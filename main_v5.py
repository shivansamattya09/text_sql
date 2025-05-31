from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List
from dotenv import load_dotenv
import os
import torch
import sqlparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

# Load tokenizer and model with proper padding configuration
tokenizer = AutoTokenizer.from_pretrained("defog/llama-3-sqlcoder-8b")
# Set pad token to eos token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "defog/llama-3-sqlcoder-8b",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your_default_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Prompt and schema setup
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:

{schema}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:
```sql
"""

DEFAULT_SCHEMA = """CREATE TABLE products (
  product_id INTEGER PRIMARY KEY,
  name VARCHAR(50),
  price DECIMAL(10,2),
  quantity INTEGER  
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY,
   name VARCHAR(50),
   address VARCHAR(100)
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY,
  name VARCHAR(50),
  region VARCHAR(50)
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY,
  product_id INTEGER,
  customer_id INTEGER,
  salesperson_id INTEGER,
  sale_date DATE,
  quantity INTEGER
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY,
  product_id INTEGER,
  supply_price DECIMAL(10,2)
);"""

CURRENT_SCHEMA = DEFAULT_SCHEMA

# Fake user database
fake_users_db = {
    "user": {
        "username": "user",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
    }
}

# Models
class User(BaseModel):
    username: str

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class SQLRequest(BaseModel):
    question: str
    schema: Optional[str] = None

class SchemaUpdate(BaseModel):
    schema_text: str

class CurrentSchemaResponse(BaseModel):
    current_schema: str

# Auth setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        return UserInDB(**db[username])
    return None

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/generate-sql")
async def generate_sql(request: SQLRequest, current_user: User = Depends(get_current_user)):
    try:
        schema = request.schema if request.schema else CURRENT_SCHEMA
        prompt = PROMPT_TEMPLATE.format(schema=schema, question=request.question)

        # Tokenize with attention mask and proper padding
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            return_attention_mask=True
        ).to(model.device)
        
        # Generate with adjusted parameters
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # More robust SQL extraction
        try:
            if "```sql" in raw_output:
                sql_query = raw_output.split("```sql")[1].split("```")[0].strip()
            else:
                sql_query = raw_output.strip()
        except IndexError:
            sql_query = raw_output.strip()

        formatted_sql = sqlparse.format(sql_query, reindent=True, indent_width=4, keyword_case='upper')

        return {
            "question": request.question,
            "sql": formatted_sql,
            "user": current_user.username
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/set-schema")
async def set_schema(update: SchemaUpdate, current_user: User = Depends(get_current_user)):
    global CURRENT_SCHEMA
    CURRENT_SCHEMA = update.schema_text
    return {"message": "Schema updated successfully"}

@app.get("/get-schema", response_model=CurrentSchemaResponse)
async def get_schema(current_user: User = Depends(get_current_user)):
    return {"current_schema": CURRENT_SCHEMA}

@app.on_event("shutdown")
def shutdown_event():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)