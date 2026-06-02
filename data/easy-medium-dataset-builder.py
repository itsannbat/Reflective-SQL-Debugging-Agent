import json
import os
import random
import re

# =====================================================================
# Database Setup (PostgreSQL)
# For the 'hard' tier to actually show bad query plans, we need realistic 
# table structures. When running the eval, inject mock data into these.
# =====================================================================

ECOMMERCE_DDL = """
CREATE TABLE Customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    country VARCHAR(50),
    signup_date DATE
);

CREATE TABLE Orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES Customers(customer_id),
    order_date DATE,
    total_amount DECIMAL(10, 2),
    status VARCHAR(20)
);

CREATE TABLE Order_Items (
    item_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES Orders(order_id),
    product_name VARCHAR(100),
    quantity INT,
    price DECIMAL(10, 2)
);
-- Note: Indexes intentionally omitted so the agent can recommend them in 'hard' mode!
"""

# =====================================================================
# Dataset Seed (The 3 Difficulty Tiers)
# =====================================================================

DATASET = [
    # ---------------------------------------------------------
    # TIER 1: EASY (Syntax Errors)
    # The agent should fix these in 1-2 rounds by reading the pg error.
    # ---------------------------------------------------------
    {
        "task_id": "easy_001",
        "difficulty": "easy",
        "database": "ecommerce_db",
        "schema_ddl": ECOMMERCE_DDL,
        "question": "Find the total number of orders placed by customers from the USA.",
        "broken_query": "SELECT COUNT(order_id) FROM Orders WHERE customer_id IN (SELCT customer_id FROM Customers WHERE country = 'USA');",
        "ground_truth_query": "SELECT COUNT(o.order_id) FROM Orders o JOIN Customers c ON o.customer_id = c.customer_id WHERE c.country = 'USA';",
        "error_type": "Typo in keyword"
    },
    {
        "task_id": "easy_002",
        "difficulty": "easy",
        "database": "ecommerce_db",
        "schema_ddl": ECOMMERCE_DDL,
        "question": "List the names of customers who have an order with a total amount greater than 1000.",
        "broken_query": "SELECT name FROM Customers c JOIN Orders o ON c.customer_id = o.customer_id WHERE o.total_amount > 1000 GROUP BY c.name",
        # Missing semicolon is fine, but let's make it a missing table alias issue
        "broken_query": "SELECT name FROM Customers JOIN Orders ON customer_id = customer_id WHERE total_amount > 1000;",
        "ground_truth_query": "SELECT DISTINCT c.name FROM Customers c JOIN Orders o ON c.customer_id = o.customer_id WHERE o.total_amount > 1000;",
        "error_type": "Ambiguous column reference"
    },

    # ---------------------------------------------------------
    # TIER 2: MEDIUM (Semantic Errors)
    # Valid SQL, but wrong logic. Agent must read data or schema to fix.
    # ---------------------------------------------------------
    {
        "task_id": "med_001",
        "difficulty": "medium",
        "database": "ecommerce_db",
        "schema_ddl": ECOMMERCE_DDL,
        "question": "Find the total revenue (sum of quantity * price) for the product 'Laptop'.",
        # Broken: Just adds price and quantity instead of multiplying, and groups by order.
        "broken_query": "SELECT SUM(quantity + price) FROM Order_Items WHERE product_name = 'Laptop';",
        "ground_truth_query": "SELECT SUM(quantity * price) FROM Order_Items WHERE product_name = 'Laptop';",
        "error_type": "Incorrect mathematical operator"
    },
    {
        "task_id": "med_002",
        "difficulty": "medium",
        "database": "ecommerce_db",
        "schema_ddl": ECOMMERCE_DDL,
        "question": "Count how many customers have NEVER placed an order.",
        # Broken: Uses an INNER JOIN, which automatically excludes customers without orders, returning 0.
        "broken_query": "SELECT COUNT(c.customer_id) FROM Customers c JOIN Orders o ON c.customer_id = o.customer_id WHERE o.order_id IS NULL;",
        "ground_truth_query": "SELECT COUNT(c.customer_id) FROM Customers c LEFT JOIN Orders o ON c.customer_id = o.customer_id WHERE o.order_id IS NULL;",
        "error_type": "Wrong JOIN type (INNER vs LEFT)"
    },

    # ---------------------------------------------------------
    # TIER 3: HARD (Performance Regressions)
    # Valid, correct logic, but terrible execution plan. 
    # Agent must use EXPLAIN ANALYZE tool to find the bottleneck.
    # ---------------------------------------------------------
    {
        "task_id": "hard_001",
        "difficulty": "hard",
        "database": "ecommerce_db",
        "schema_ddl": ECOMMERCE_DDL,
        "question": "Get the latest order date for every customer.",
        # Broken: Correlated subquery in the SELECT clause. Causes O(N^2) sequential scans.
        "broken_query": "SELECT c.name, (SELECT MAX(order_date) FROM Orders o WHERE o.customer_id = c.customer_id) as latest_order FROM Customers c;",
        "ground_truth_query": "SELECT c.name, MAX(o.order_date) FROM Customers c LEFT JOIN Orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.name;",
        "error_type": "Correlated subquery / N+1 query problem"
    },
    {
        "task_id": "hard_002",
        "difficulty": "hard",
        "database": "ecommerce_db",
        "schema_ddl": ECOMMERCE_DDL,
        "question": "Find all orders placed in 2023.",
        # Broken: Wrapping the column in a function prevents index usage (Index Scan -> Seq Scan).
        "broken_query": "SELECT * FROM Orders WHERE EXTRACT(YEAR FROM order_date) = 2023;",
        "ground_truth_query": "SELECT * FROM Orders WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01';",
        "error_type": "Function on indexed column (Sargability violation)"
    }
]

# =====================================================================
# Perturbation Functions for Spider Data (Easy & Medium)
# =====================================================================

def perturb_easy(query):
    """Applies syntax-breaking manipulations."""
    options = []
    if '(' in query or ')' in query: options.append('parenthesis')
    if re.search(r'\bGROUP BY\b', query, re.IGNORECASE): options.append('groupby')
    if re.search(r'\bSELECT\b', query, re.IGNORECASE): options.append('select')
    
    if not options:
        return query, "None"
        
    choice = random.choice(options)
    
    if choice == 'parenthesis':
        chars = list(query)
        paren_indices = [i for i, c in enumerate(chars) if c in '()']
        chars.pop(random.choice(paren_indices))
        return "".join(chars), "Removed random parenthesis"
        
    elif choice == 'groupby':
        # Removes GROUP BY and the columns up to the next clause or end of string
        new_query = re.sub(r'\s+GROUP BY\s+.*?(?=\s+(?:HAVING|ORDER BY|LIMIT)|$)', '', query, flags=re.IGNORECASE)
        return new_query, "Deleted GROUP BY clause"
        
    elif choice == 'select':
        new_query = re.sub(r'\bSELECT\b', 'SELET', query, count=1, flags=re.IGNORECASE)
        return new_query, "Misspelled SELECT as SELET"
        
def perturb_medium(query):
    """Applies semantic/logic-breaking manipulations using regex."""
    options = []
    if re.search(r'\b(?:INNER )?JOIN\b|\bLEFT JOIN\b', query, re.IGNORECASE): options.append('join')
    if re.search(r'\bMAX\(|\bMIN\(', query, re.IGNORECASE): options.append('agg')
    if re.search(r'\bAND\b|\bOR\b', query, re.IGNORECASE): options.append('logic')
    
    if not options:
        return query, "None"
        
    choice = random.choice(options)
    
    if choice == 'join':
        if re.search(r'\bLEFT JOIN\b', query, re.IGNORECASE):
            return re.sub(r'\bLEFT JOIN\b', 'INNER JOIN', query, count=1, flags=re.IGNORECASE), "Swapped LEFT JOIN to INNER JOIN"
        else:
            return re.sub(r'\b(?:INNER )?JOIN\b', 'LEFT JOIN', query, count=1, flags=re.IGNORECASE), "Swapped INNER JOIN to LEFT JOIN"
            
    elif choice == 'agg':
        if re.search(r'\bMAX\(', query, re.IGNORECASE):
            return re.sub(r'\bMAX\(', 'MIN(', query, count=1, flags=re.IGNORECASE), "Swapped MAX() for MIN()"
        else:
            return re.sub(r'\bMIN\(', 'MAX(', query, count=1, flags=re.IGNORECASE), "Swapped MIN() for MAX()"
            
    elif choice == 'logic':
        if re.search(r'\bAND\b', query, re.IGNORECASE):
            return re.sub(r'\bAND\b', 'OR', query, count=1, flags=re.IGNORECASE), "Changed AND to OR"
        else:
            return re.sub(r'\bOR\b', 'AND', query, count=1, flags=re.IGNORECASE), "Changed OR to AND"

def expand_dataset_with_spider(spider_data_path=None):
    """Loads Spider dev data (or uses mock data) and generates Easy/Medium tiers."""
    
    # Mock Spider queries for immediate testing without the json file
    mock_spider_queries = [
        {"query": "SELECT MAX(age) FROM student WHERE city = 'NY' AND gender = 'F'", "question": "What is the max age of female students in NY?"},
        {"query": "SELECT name FROM employee INNER JOIN department ON employee.dept_id = department.id GROUP BY department.id", "question": "List employee names grouped by department."},
        {"query": "SELECT MIN(salary) FROM instructor WHERE dept_name = 'Comp. Sci.' OR dept_name = 'Physics'", "question": "What is the minimum salary in CS or Physics?"},
        {"query": "SELECT T1.name FROM driver AS T1 JOIN races AS T2 ON T1.driverid = T2.driverid", "question": "Find names of drivers who have races."},
        {"query": "SELECT count(*) FROM cars_data WHERE cylinders = 8 AND year < 1980", "question": "How many 8-cylinder cars are from before 1980?"}
    ] * 10 # Duplicate to simulate a larger dataset pool
    
    # When you have the actual file, you can pass the path and this will override the mock:
    if spider_data_path and os.path.exists(spider_data_path):
        with open(spider_data_path, 'r') as f:
            mock_spider_queries = json.load(f)
            
    # Generate 15 Easy Tier Queries
    easy_count = 0
    for item in mock_spider_queries:
        if easy_count >= 15: break
        broken, error_type = perturb_easy(item["query"])
        if error_type != "None":
            DATASET.append({
                "task_id": f"spider_easy_{easy_count+1}",
                "difficulty": "easy",
                "database": "spider_mock_db",
                "schema_ddl": "-- (Target Spider Schema would be injected here)",
                "question": item["question"],
                "broken_query": broken,
                "ground_truth_query": item["query"],
                "error_type": error_type
            })
            easy_count += 1
            
    # Generate 15 Medium Tier Queries
    med_count = 0
    # Reverse iteration to ensure we grab different queries from the Easy set
    for item in reversed(mock_spider_queries):
        if med_count >= 15: break
        broken, error_type = perturb_medium(item["query"])
        if error_type != "None":
            DATASET.append({
                "task_id": f"spider_med_{med_count+1}",
                "difficulty": "medium",
                "database": "spider_mock_db",
                "schema_ddl": "-- (Target Spider Schema would be injected here)",
                "question": item["question"],
                "broken_query": broken,
                "ground_truth_query": item["query"],
                "error_type": error_type
            })
            med_count += 1

def export_dataset(filepath="dataset.json"):
    """Writes the dataset to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(DATASET, f, indent=4)
    print(f"✅ Successfully generated dataset seed with {len(DATASET)} queries at {filepath}")

if __name__ == "__main__":
    expand_dataset_with_spider()
    export_dataset("data/dataset.json")