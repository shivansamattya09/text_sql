# model_handler.py
from transformers import AutoTokenizer
import torch
import pickle

# Load tokenizer and model
model_name = "defog/llama-3-sqlcoder-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# SQL prompt template
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Generate a SQL query to answer this question: `{question}`

DDL statements:

CREATE TABLE products (
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
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The following SQL query best answers the question `{question}`:
```sql
"""

def generate_query(question: str) -> str:
    formatted_prompt = prompt.format(question=question)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,
            do_sample=False,
            temperature=0.0,
            top_p=1,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    try:
        sql_part = decoded.split("```sql")[1].split(";")[0]
    except IndexError:
        sql_part = decoded  # fallback if format is off

    return sql_part
