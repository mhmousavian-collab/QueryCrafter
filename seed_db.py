#!/app/venv/bin/python
import os, time, logging, random
import psycopg2
from psycopg2 import sql
from faker import Faker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed_db")
fake = Faker()


def get_conn(retries=60, backoff=1.0):
    dbname = os.getenv("POSTGRES_DB", "pg_db")
    user = os.getenv("POSTGRES_USER", "user")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("PGHOST", None)
    port = os.getenv("PGPORT", "5432")
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(
                dbname=dbname, user=user, password=password, host=host, port=port
            )
            logger.info("Connected to Postgres (attempt %d)", attempt)
            return conn
        except Exception as ex:
            logger.warning("Connection attempt %d failed: %s", attempt, ex)
            time.sleep(backoff)
            backoff = min(backoff * 1.25, 5.0)
    raise RuntimeError("Unable to connect to Postgres")


def table_exists(cur, name):
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name=%s);",
        (name,),
    )
    return cur.fetchone()[0]


def row_count(cur, name):
    cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(name)))
    return cur.fetchone()[0]


def create_schema(cur):
    cur.execute(
        """CREATE TABLE IF NOT EXISTS customers(
        id SERIAL PRIMARY KEY, name TEXT, email TEXT, registration_date DATE);"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS products(
        id SERIAL PRIMARY KEY, name TEXT, category TEXT, price NUMERIC);"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS orders(
        id SERIAL PRIMARY KEY, customer_id INT REFERENCES customers(id),
        product_id INT REFERENCES products(id), order_date DATE,
        quantity INT, status TEXT);"""
    )


def populate(cur):
    customer_ids = []
    for _ in range(300):
        cur.execute(
            "INSERT INTO customers(name,email,registration_date) VALUES(%s,%s,%s) RETURNING id;",
            (
                fake.name(),
                fake.email(),
                fake.date_between(start_date="-1y", end_date="today"),
            ),
        )
        customer_ids.append(cur.fetchone()[0])
    product_ids = []
    for _ in range(50):
        cur.execute(
            "INSERT INTO products(name,category,price) VALUES(%s,%s,%s) RETURNING id;",
            (
                fake.word().capitalize(),
                random.choice(["Books", "Electronics", "Clothing", "Home"]),
                round(random.uniform(10, 500), 2),
            ),
        )
        product_ids.append(cur.fetchone()[0])
    for _ in range(1000):
        cur.execute(
            "INSERT INTO orders(customer_id,product_id,order_date,quantity,status) VALUES(%s,%s,%s,%s,%s);",
            (
                random.choice(customer_ids),
                random.choice(product_ids),
                fake.date_between(start_date="-6mo", end_date="today"),
                random.randint(1, 5),
                random.choice(["pending", "shipped", "delivered", "cancelled"]),
            ),
        )


def main():
    conn = get_conn()
    cur = conn.cursor()
    create_schema(cur)
    conn.commit()
    if table_exists(cur, "customers") and row_count(cur, "customers") > 0:
        logger.info("Database already populated, skipping seeding.")
    else:
        logger.info("Populating database...")
        populate(cur)
        conn.commit()
        logger.info("Seeding complete.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
