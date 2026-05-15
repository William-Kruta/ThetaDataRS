import polars as pl
import duckdb

from .config import get_db_path


def get_connection(db_path: str = None) -> duckdb.DuckDBPyConnection:
    if db_path is None:
        db_path = str(get_db_path())
    conn = duckdb.connect(db_path)
    _init_tables(conn)
    return conn


def _init_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS option_symbols (
            symbol      TEXT      NOT NULL,
            fetched_at  TIMESTAMP NOT NULL,
            PRIMARY KEY (symbol)
        );

        CREATE TABLE IF NOT EXISTS option_expirations (
            root        TEXT      NOT NULL,
            expiration  DATE      NOT NULL,
            fetched_at  TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration)
        );

        CREATE TABLE IF NOT EXISTS option_contracts (
            root        TEXT      NOT NULL,
            expiration  DATE      NOT NULL,
            strike      DOUBLE    NOT NULL,
            "right"     TEXT      NOT NULL,
            as_of_date  DATE      NOT NULL,
            fetched_at  TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS option_dates (
            root         TEXT      NOT NULL,
            expiration   DATE      NOT NULL,
            date         DATE      NOT NULL,
            request_type TEXT      NOT NULL,
            fetched_at   TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, date, request_type)
        );

        CREATE TABLE IF NOT EXISTS option_eod (
            date          DATE      NOT NULL,
            root          TEXT      NOT NULL,
            expiration    DATE      NOT NULL,
            strike        DOUBLE    NOT NULL,
            "right"       TEXT      NOT NULL,
            created       TIMESTAMP,
            last_trade    TIMESTAMP,
            "open"        DOUBLE,
            high          DOUBLE,
            low           DOUBLE,
            "close"       DOUBLE,
            volume        BIGINT,
            count         BIGINT,
            bid_size      INTEGER,
            bid_exchange  INTEGER,
            bid           DOUBLE,
            bid_condition INTEGER,
            ask_size      INTEGER,
            ask_exchange  INTEGER,
            ask           DOUBLE,
            ask_condition INTEGER,
            fetched_at    TIMESTAMP NOT NULL,
            PRIMARY KEY (date, root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS option_ohlcv (
            date        DATE      NOT NULL,
            root        TEXT      NOT NULL,
            expiration  DATE      NOT NULL,
            strike      DOUBLE    NOT NULL,
            "right"     TEXT      NOT NULL,
            timestamp   TIMESTAMP NOT NULL,
            "open"      DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            "close"     DOUBLE,
            volume      BIGINT,
            count       BIGINT,
            fetched_at  TIMESTAMP NOT NULL,
            PRIMARY KEY (date, timestamp, root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS option_trades (
            date           DATE      NOT NULL,
            root           TEXT      NOT NULL,
            expiration     DATE      NOT NULL,
            strike         DOUBLE    NOT NULL,
            "right"        TEXT      NOT NULL,
            timestamp      TIMESTAMP NOT NULL,
            sequence       BIGINT,
            ext_condition1 INTEGER,
            ext_condition2 INTEGER,
            ext_condition3 INTEGER,
            ext_condition4 INTEGER,
            condition      INTEGER,
            size           INTEGER,
            exchange       INTEGER,
            price          DOUBLE,
            fetched_at     TIMESTAMP NOT NULL,
            PRIMARY KEY (date, timestamp, root, expiration, strike, "right", sequence)
        );

        CREATE TABLE IF NOT EXISTS option_quotes (
            date          DATE      NOT NULL,
            root          TEXT      NOT NULL,
            expiration    DATE      NOT NULL,
            strike        DOUBLE    NOT NULL,
            "right"       TEXT      NOT NULL,
            timestamp     TIMESTAMP NOT NULL,
            bid           DOUBLE,
            ask           DOUBLE,
            bid_size      INTEGER,
            ask_size      INTEGER,
            bid_exchange  INTEGER,
            ask_exchange  INTEGER,
            bid_condition INTEGER,
            ask_condition INTEGER,
            fetched_at    TIMESTAMP NOT NULL,
            PRIMARY KEY (date, timestamp, root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS option_trade_quotes (
            date           DATE      NOT NULL,
            root           TEXT      NOT NULL,
            expiration     DATE      NOT NULL,
            strike         DOUBLE    NOT NULL,
            "right"        TEXT      NOT NULL,
            trade_timestamp TIMESTAMP NOT NULL,
            quote_timestamp TIMESTAMP,
            sequence       BIGINT,
            ext_condition1 INTEGER,
            ext_condition2 INTEGER,
            ext_condition3 INTEGER,
            ext_condition4 INTEGER,
            condition      INTEGER,
            size           INTEGER,
            exchange       INTEGER,
            price          DOUBLE,
            bid_size       INTEGER,
            bid_exchange   INTEGER,
            bid            DOUBLE,
            bid_condition  INTEGER,
            ask_size       INTEGER,
            ask_exchange   INTEGER,
            ask            DOUBLE,
            ask_condition  INTEGER,
            fetched_at     TIMESTAMP NOT NULL,
            PRIMARY KEY (date, trade_timestamp, root, expiration, strike, "right", sequence)
        );

        CREATE TABLE IF NOT EXISTS option_open_interest (
            date          DATE      NOT NULL,
            root          TEXT      NOT NULL,
            expiration    DATE      NOT NULL,
            strike        DOUBLE    NOT NULL,
            "right"       TEXT      NOT NULL,
            timestamp     TIMESTAMP,
            open_interest BIGINT,
            fetched_at    TIMESTAMP NOT NULL,
            PRIMARY KEY (date, root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS option_greeks (
            date               DATE      NOT NULL,
            root               TEXT      NOT NULL,
            expiration         DATE      NOT NULL,
            strike             DOUBLE    NOT NULL,
            "right"            TEXT      NOT NULL,
            timestamp          TIMESTAMP NOT NULL,
            bid                DOUBLE,
            ask                DOUBLE,
            delta              DOUBLE,
            vega               DOUBLE,
            theta              DOUBLE,
            rho                DOUBLE,
            epsilon            DOUBLE,
            "lambda"           DOUBLE,
            gamma              DOUBLE,
            vanna              DOUBLE,
            charm              DOUBLE,
            vomma              DOUBLE,
            veta               DOUBLE,
            vera               DOUBLE,
            speed              DOUBLE,
            zomma              DOUBLE,
            color              DOUBLE,
            ultima             DOUBLE,
            d1                 DOUBLE,
            d2                 DOUBLE,
            dual_delta         DOUBLE,
            dual_gamma         DOUBLE,
            implied_vol        DOUBLE,
            iv_error           DOUBLE,
            underlying_timestamp TIMESTAMP,
            underlying_price   DOUBLE,
            fetched_at         TIMESTAMP NOT NULL,
            PRIMARY KEY (date, timestamp, root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS option_greeks_eod (
            date               DATE      NOT NULL,
            root               TEXT      NOT NULL,
            expiration         DATE      NOT NULL,
            strike             DOUBLE    NOT NULL,
            "right"            TEXT      NOT NULL,
            timestamp          TIMESTAMP,
            "open"             DOUBLE,
            high               DOUBLE,
            low                DOUBLE,
            "close"            DOUBLE,
            volume             BIGINT,
            count              BIGINT,
            bid_size           INTEGER,
            bid_exchange       INTEGER,
            bid                DOUBLE,
            bid_condition      INTEGER,
            ask_size           INTEGER,
            ask_exchange       INTEGER,
            ask                DOUBLE,
            ask_condition      INTEGER,
            delta              DOUBLE,
            theta              DOUBLE,
            vega               DOUBLE,
            rho                DOUBLE,
            epsilon            DOUBLE,
            "lambda"           DOUBLE,
            gamma              DOUBLE,
            vanna              DOUBLE,
            charm              DOUBLE,
            vomma              DOUBLE,
            veta               DOUBLE,
            vera               DOUBLE,
            speed              DOUBLE,
            zomma              DOUBLE,
            color              DOUBLE,
            ultima             DOUBLE,
            d1                 DOUBLE,
            d2                 DOUBLE,
            dual_delta         DOUBLE,
            dual_gamma         DOUBLE,
            implied_vol        DOUBLE,
            iv_error           DOUBLE,
            underlying_timestamp TIMESTAMP,
            underlying_price   DOUBLE,
            fetched_at         TIMESTAMP NOT NULL,
            PRIMARY KEY (date, root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS option_trade_greeks (
            date               DATE      NOT NULL,
            root               TEXT      NOT NULL,
            expiration         DATE      NOT NULL,
            strike             DOUBLE    NOT NULL,
            "right"            TEXT      NOT NULL,
            timestamp          TIMESTAMP NOT NULL,
            sequence           BIGINT,
            ext_condition1     INTEGER,
            ext_condition2     INTEGER,
            ext_condition3     INTEGER,
            ext_condition4     INTEGER,
            condition          INTEGER,
            size               INTEGER,
            exchange           INTEGER,
            price              DOUBLE,
            delta              DOUBLE,
            theta              DOUBLE,
            vega               DOUBLE,
            rho                DOUBLE,
            epsilon            DOUBLE,
            "lambda"           DOUBLE,
            gamma              DOUBLE,
            vanna              DOUBLE,
            charm              DOUBLE,
            vomma              DOUBLE,
            veta               DOUBLE,
            vera               DOUBLE,
            speed              DOUBLE,
            zomma              DOUBLE,
            color              DOUBLE,
            ultima             DOUBLE,
            d1                 DOUBLE,
            d2                 DOUBLE,
            dual_delta         DOUBLE,
            dual_gamma         DOUBLE,
            implied_vol        DOUBLE,
            iv_error           DOUBLE,
            underlying_timestamp TIMESTAMP,
            underlying_price   DOUBLE,
            fetched_at         TIMESTAMP NOT NULL,
            PRIMARY KEY (date, timestamp, root, expiration, strike, "right", sequence)
        );

        CREATE TABLE IF NOT EXISTS snapshot_ohlcv (
            root        TEXT      NOT NULL,
            expiration  DATE      NOT NULL,
            strike      DOUBLE    NOT NULL,
            "right"     TEXT      NOT NULL,
            timestamp   TIMESTAMP,
            "open"      DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            "close"     DOUBLE,
            volume      BIGINT,
            count       BIGINT,
            fetched_at  TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS snapshot_trade (
            root        TEXT      NOT NULL,
            expiration  DATE      NOT NULL,
            strike      DOUBLE    NOT NULL,
            "right"     TEXT      NOT NULL,
            timestamp   TIMESTAMP,
            sequence    BIGINT,
            condition   INTEGER,
            size        INTEGER,
            exchange    INTEGER,
            price       DOUBLE,
            fetched_at  TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS snapshot_quote (
            root          TEXT      NOT NULL,
            expiration    DATE      NOT NULL,
            strike        DOUBLE    NOT NULL,
            "right"       TEXT      NOT NULL,
            timestamp     TIMESTAMP,
            bid_size      INTEGER,
            bid_exchange  INTEGER,
            bid           DOUBLE,
            bid_condition INTEGER,
            ask_size      INTEGER,
            ask_exchange  INTEGER,
            ask           DOUBLE,
            ask_condition INTEGER,
            fetched_at    TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS snapshot_open_interest (
            root          TEXT      NOT NULL,
            expiration    DATE      NOT NULL,
            strike        DOUBLE    NOT NULL,
            "right"       TEXT      NOT NULL,
            timestamp     TIMESTAMP,
            open_interest BIGINT,
            fetched_at    TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS snapshot_market_value (
            root         TEXT      NOT NULL,
            expiration   DATE      NOT NULL,
            strike       DOUBLE    NOT NULL,
            "right"      TEXT      NOT NULL,
            timestamp    TIMESTAMP,
            market_bid   DOUBLE,
            market_ask   DOUBLE,
            market_price DOUBLE,
            fetched_at   TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS snapshot_implied_volatility (
            root                 TEXT      NOT NULL,
            expiration           DATE      NOT NULL,
            strike               DOUBLE    NOT NULL,
            "right"              TEXT      NOT NULL,
            timestamp            TIMESTAMP,
            bid                  DOUBLE,
            ask                  DOUBLE,
            implied_vol          DOUBLE,
            iv_error             TEXT,
            underlying_timestamp TIMESTAMP,
            underlying_price     DOUBLE,
            fetched_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS snapshot_greeks_first_order (
            root                 TEXT      NOT NULL,
            expiration           DATE      NOT NULL,
            strike               DOUBLE    NOT NULL,
            "right"              TEXT      NOT NULL,
            timestamp            TIMESTAMP,
            bid                  DOUBLE,
            ask                  DOUBLE,
            delta                DOUBLE,
            theta                DOUBLE,
            vega                 DOUBLE,
            rho                  DOUBLE,
            epsilon              DOUBLE,
            "lambda"             DOUBLE,
            implied_vol          DOUBLE,
            iv_error             TEXT,
            underlying_timestamp TIMESTAMP,
            underlying_price     DOUBLE,
            fetched_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS snapshot_greeks_second_order (
            root                 TEXT      NOT NULL,
            expiration           DATE      NOT NULL,
            strike               DOUBLE    NOT NULL,
            "right"              TEXT      NOT NULL,
            timestamp            TIMESTAMP,
            bid                  DOUBLE,
            ask                  DOUBLE,
            gamma                DOUBLE,
            vanna                DOUBLE,
            charm                DOUBLE,
            vomma                DOUBLE,
            veta                 DOUBLE,
            implied_vol          DOUBLE,
            iv_error             TEXT,
            underlying_timestamp TIMESTAMP,
            underlying_price     DOUBLE,
            fetched_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS snapshot_greeks_third_order (
            root                 TEXT      NOT NULL,
            expiration           DATE      NOT NULL,
            strike               DOUBLE    NOT NULL,
            "right"              TEXT      NOT NULL,
            timestamp            TIMESTAMP,
            bid                  DOUBLE,
            ask                  DOUBLE,
            speed                DOUBLE,
            zomma                DOUBLE,
            color                DOUBLE,
            ultima               DOUBLE,
            implied_vol          DOUBLE,
            iv_error             TEXT,
            underlying_timestamp TIMESTAMP,
            underlying_price     DOUBLE,
            fetched_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS snapshot_greeks_all (
            root                 TEXT      NOT NULL,
            expiration           DATE      NOT NULL,
            strike               DOUBLE    NOT NULL,
            "right"              TEXT      NOT NULL,
            timestamp            TIMESTAMP,
            bid                  DOUBLE,
            ask                  DOUBLE,
            delta                DOUBLE,
            theta                DOUBLE,
            vega                 DOUBLE,
            rho                  DOUBLE,
            epsilon              DOUBLE,
            "lambda"             DOUBLE,
            gamma                DOUBLE,
            vanna                DOUBLE,
            charm                DOUBLE,
            vomma                DOUBLE,
            veta                 DOUBLE,
            speed                DOUBLE,
            zomma                DOUBLE,
            color                DOUBLE,
            ultima               DOUBLE,
            implied_vol          DOUBLE,
            iv_error             TEXT,
            underlying_timestamp TIMESTAMP,
            underlying_price     DOUBLE,
            fetched_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (root, expiration, strike, "right")
        );

        CREATE TABLE IF NOT EXISTS cache_fetches (
            fetch_id         TEXT      NOT NULL,
            endpoint         TEXT      NOT NULL,
            root             TEXT      NOT NULL,
            params_hash      TEXT      NOT NULL,
            params_json      TEXT      NOT NULL,
            coverage_json    TEXT      NOT NULL,
            cache_policy     TEXT      NOT NULL,
            status           TEXT      NOT NULL,
            row_count        BIGINT,
            fetched_at       TIMESTAMP NOT NULL,
            duration_seconds DOUBLE,
            error_type       TEXT,
            error_message    TEXT,
            PRIMARY KEY (fetch_id)
        );

        CREATE INDEX IF NOT EXISTS idx_cache_fetches_lookup
            ON cache_fetches(endpoint, root, status, fetched_at);
    """)


def insert_data(
    df: pl.DataFrame,
    table_name: str,
    conn: duckdb.DuckDBPyConnection,
    columns: list[str] = None,
) -> None:
    if df.is_empty():
        return

    if columns:
        df = df.select([c for c in columns if c in df.columns])

    col_names = ", ".join(df.columns)
    conn.execute(f"""
        INSERT OR IGNORE INTO {table_name} ({col_names})
        SELECT {col_names} FROM df
    """)
