import json
import os
import random
import re
import argparse

# =====================================================================
# Database Setup (PostgreSQL) - Hard Tier
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
# Hard Tier Seed (Hand-crafted performance cases)
# 
# TODO: Add more hand-crafted entries here that target specific performance anti-patterns
# =====================================================================

DATASET = [
    {
        "task_id": "hard_001",
        "difficulty": "hard",
        "database": "ecommerce_db",
        "schema_ddl": ECOMMERCE_DDL,
        "question": "Get the latest order date for every customer.",
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
        "broken_query": "SELECT * FROM Orders WHERE EXTRACT(YEAR FROM order_date) = 2023;",
        "ground_truth_query": "SELECT * FROM Orders WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01';",
        "error_type": "Function on indexed column (Sargability violation)"
    }
]

# =====================================================================
# Schema DDL Builder from Spider tables.json
# =====================================================================

def build_ddl_from_spider_schema(db_entry: dict) -> str:
    """
    Converts a Spider tables.json db entry into a CREATE TABLE DDL string.
    
    Spider schema shape:
      {
        "db_id": "concert_singer",
        "table_names_original": ["stadium", "singer", ...],
        "column_names_original": [[-1, "*"], [0, "Stadium_ID"], [0, "Location"], ...],
        "column_types": ["text", "number", "text", ...],
        "primary_keys": [1, 5, ...],          # indices into column_names_original
        "foreign_keys": [[3, 8], [7, 1], ...]  # pairs of column indices
      }
    """
    table_names = db_entry.get("table_names_original", [])
    col_names = db_entry.get("column_names_original", [])   # [table_idx, col_name]
    col_types = db_entry.get("column_types", [])
    primary_keys = set(db_entry.get("primary_keys", []))

    # Map Spider types -> rough SQL types
    type_map = {
        "number": "NUMERIC",
        "text": "TEXT",
        "boolean": "BOOLEAN",
        "time": "DATE",
        "others": "TEXT",
    }

    # Group columns by table index (skip index 0 which is the wildcard "*")
    tables: dict[int, list] = {i: [] for i in range(len(table_names))}
    for col_idx, (table_idx, col_name) in enumerate(col_names):
        if table_idx == -1:
            continue  # skip the global "*" column
        sql_type = type_map.get(col_types[col_idx], "TEXT")
        is_pk = col_idx in primary_keys
        tables[table_idx].append((col_name, sql_type, is_pk))

    # Build foreign key lookup: col_idx -> (table_name, col_name)
    fk_map: dict[int, tuple] = {}
    for from_col, to_col in db_entry.get("foreign_keys", []):
        to_table_idx, to_col_name = col_names[to_col]
        fk_map[from_col] = (table_names[to_table_idx], to_col_name)

    ddl_statements = []
    for table_idx, table_name in enumerate(table_names):
        cols = tables.get(table_idx, [])
        if not cols:
            continue

        col_defs = []
        fk_constraints = []

        for col_idx, (col_name, sql_type, is_pk) in enumerate(cols):
            # Recover the original column index in col_names for FK lookup
            original_col_idx = next(
                i for i, (ti, cn) in enumerate(col_names)
                if ti == table_idx and cn == col_name
            )
            pk_str = " PRIMARY KEY" if is_pk else ""
            col_defs.append(f"    {col_name} {sql_type}{pk_str}")

            if original_col_idx in fk_map:
                ref_table, ref_col = fk_map[original_col_idx]
                fk_constraints.append(
                    f"    FOREIGN KEY ({col_name}) REFERENCES {ref_table}({ref_col})"
                )

        all_defs = col_defs + fk_constraints
        ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(all_defs) + "\n);"
        ddl_statements.append(ddl)

    return "\n\n".join(ddl_statements)


def load_spider_schemas(tables_path: str) -> dict[str, str]:
    """Returns a dict of {db_id -> DDL string} from Spider's tables.json."""
    if not tables_path or not os.path.exists(tables_path):
        return {}
    with open(tables_path, "r") as f:
        tables_data = json.load(f)
    return {db["db_id"]: build_ddl_from_spider_schema(db) for db in tables_data}

# =====================================================================
# Perturbation Functions
# =====================================================================

def perturb_easy(query: str) -> tuple[str, str | None]:
    """Applies syntax-breaking manipulations. Returns (broken_query, error_type)."""
    options = []
    if "(" in query or ")" in query:
        options.append("parenthesis")
    if re.search(r"\bGROUP BY\b", query, re.IGNORECASE):
        options.append("groupby")
    if re.search(r"\bSELECT\b", query, re.IGNORECASE):
        options.append("select")

    if not options:
        return query, None

    choice = random.choice(options)

    if choice == "parenthesis":
        chars = list(query)
        paren_indices = [i for i, c in enumerate(chars) if c in "()"]
        chars.pop(random.choice(paren_indices))
        return "".join(chars), "Removed random parenthesis"

    elif choice == "groupby":
        new_query = re.sub(
            r"\s+GROUP BY\s+.*?(?=\s+(?:HAVING|ORDER BY|LIMIT)|$)",
            "",
            query,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return new_query, "Deleted GROUP BY clause"

    elif choice == "select":
        new_query = re.sub(r"\bSELECT\b", "SELET", query, count=1, flags=re.IGNORECASE)
        return new_query, "Misspelled SELECT as SELET"


def perturb_medium(query: str) -> tuple[str, str | None]:
    """Applies semantic/logic-breaking manipulations. Returns (broken_query, error_type)."""
    options = []
    if re.search(r"\b(?:INNER )?JOIN\b|\bLEFT JOIN\b", query, re.IGNORECASE):
        options.append("join")
    if re.search(r"\bMAX\(|\bMIN\(", query, re.IGNORECASE):
        options.append("agg")
    if re.search(r"\bAND\b|\bOR\b", query, re.IGNORECASE):
        options.append("logic")

    if not options:
        return query, None

    choice = random.choice(options)

    if choice == "join":
        if re.search(r"\bLEFT JOIN\b", query, re.IGNORECASE):
            return (
                re.sub(r"\bLEFT JOIN\b", "INNER JOIN", query, count=1, flags=re.IGNORECASE),
                "Swapped LEFT JOIN to INNER JOIN",
            )
        else:
            return (
                re.sub(r"\b(?:INNER )?JOIN\b", "LEFT JOIN", query, count=1, flags=re.IGNORECASE),
                "Swapped INNER JOIN to LEFT JOIN",
            )

    elif choice == "agg":
        if re.search(r"\bMAX\(", query, re.IGNORECASE):
            return (
                re.sub(r"\bMAX\(", "MIN(", query, count=1, flags=re.IGNORECASE),
                "Swapped MAX() for MIN()",
            )
        else:
            return (
                re.sub(r"\bMIN\(", "MAX(", query, count=1, flags=re.IGNORECASE),
                "Swapped MIN() for MAX()",
            )

    elif choice == "logic":
        if re.search(r"\bAND\b", query, re.IGNORECASE):
            return (
                re.sub(r"\bAND\b", "OR", query, count=1, flags=re.IGNORECASE),
                "Changed AND to OR",
            )
        else:
            return (
                re.sub(r"\bOR\b", "AND", query, count=1, flags=re.IGNORECASE),
                "Changed OR to AND",
            )

# =====================================================================
# Spider Expansion
# =====================================================================

MOCK_SPIDER_QUERIES = [
    {"query": "SELECT MAX(age) FROM student WHERE city = 'NY' AND gender = 'F'",  "question": "What is the max age of female students in NY?",         "db_id": "mock_db"},
    {"query": "SELECT name FROM employee INNER JOIN department ON employee.dept_id = department.id GROUP BY department.id", "question": "List employee names grouped by department.", "db_id": "mock_db"},
    {"query": "SELECT MIN(salary) FROM instructor WHERE dept_name = 'Comp. Sci.' OR dept_name = 'Physics'", "question": "What is the minimum salary in CS or Physics?", "db_id": "mock_db"},
    {"query": "SELECT T1.name FROM driver AS T1 JOIN races AS T2 ON T1.driverid = T2.driverid", "question": "Find names of drivers who have races.",    "db_id": "mock_db"},
    {"query": "SELECT count(*) FROM cars_data WHERE cylinders = 8 AND year < 1980", "question": "How many 8-cylinder cars are from before 1980?",       "db_id": "mock_db"},
] * 10


def expand_dataset_with_spider(
    spider_data_path: str | None = None,
    spider_tables_path: str | None = None,
    easy_target: int = 15,
    medium_target: int = 15,
) -> None:
    """
    Loads Spider dev.json (falls back to mock data) and appends
    easy + medium perturbation entries to DATASET.
    """
    # Load schemas (empty dict if no tables.json provided)
    schemas = load_spider_schemas(spider_tables_path)

    # Load queries
    if spider_data_path and os.path.exists(spider_data_path):
        with open(spider_data_path, "r") as f:
            spider_queries = json.load(f)
        print(f"📂 Loaded {len(spider_queries)} queries from {spider_data_path}")
    else:
        spider_queries = MOCK_SPIDER_QUERIES
        print("⚠️  No Spider dev.json found — using mock queries.")

    # --- Easy tier ---
    easy_count = 0
    for item in spider_queries:
        if easy_count >= easy_target:
            break
        broken, error_type = perturb_easy(item["query"])
        if error_type is None:
            continue

        db_id = item.get("db_id", "spider_db")
        DATASET.append({
            "task_id": f"spider_easy_{easy_count + 1:03d}",
            "difficulty": "easy",
            "database": db_id,
            "schema_ddl": schemas.get(db_id, f"-- Schema for '{db_id}' not available"),
            "question": item["question"],
            "broken_query": broken,
            "ground_truth_query": item["query"],
            "error_type": error_type,
        })
        easy_count += 1

    print(f"✅ Generated {easy_count} easy entries")

    # --- Medium tier (iterate in reverse to avoid overlap with easy) ---
    med_count = 0
    for item in reversed(spider_queries):
        if med_count >= medium_target:
            break
        broken, error_type = perturb_medium(item["query"])
        if error_type is None:
            continue

        db_id = item.get("db_id", "spider_db")
        DATASET.append({
            "task_id": f"spider_med_{med_count + 1:03d}",
            "difficulty": "medium",
            "database": db_id,
            "schema_ddl": schemas.get(db_id, f"-- Schema for '{db_id}' not available"),
            "question": item["question"],
            "broken_query": broken,
            "ground_truth_query": item["query"],
            "error_type": error_type,
        })
        med_count += 1

    print(f"✅ Generated {med_count} medium entries")

# =====================================================================
# Export
# =====================================================================

def export_dataset(filepath: str = "data/dataset.json") -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(DATASET, f, indent=4)
    print(f"✅ Exported {len(DATASET)} total entries → {filepath}")


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the SQL debugging eval dataset.")
    parser.add_argument(
        "--spider_dev",
        default=None,
        help="Path to Spider dev.json (e.g. spider/dev.json)",
    )
    parser.add_argument(
        "--spider_tables",
        default=None,
        help="Path to Spider tables.json (e.g. spider/tables.json)",
    )
    parser.add_argument(
        "--easy_target",
        type=int,
        default=15,
        help="Number of easy entries to generate from Spider (default: 15)",
    )
    parser.add_argument(
        "--medium_target",
        type=int,
        default=15,
        help="Number of medium entries to generate from Spider (default: 15)",
    )
    parser.add_argument(
        "--output",
        default="data/dataset.json",
        help="Output path for the dataset JSON (default: data/dataset.json)",
    )
    args = parser.parse_args()

    expand_dataset_with_spider(
        spider_data_path=args.spider_dev,
        spider_tables_path=args.spider_tables,
        easy_target=args.easy_target,
        medium_target=args.medium_target,
    )
    export_dataset(args.output)