import argparse
import datetime as dt
import inspect

from thetadatars import ThetaDataRS


def show_available_helpers(theta: ThetaDataRS, limit: int = 20) -> None:
    """Print a quick summary of methods exposed by the ThetaDataRS facade."""
    public_methods = [
        name
        for name in dir(theta)
        if name.startswith(
            ("fetch_", "get_", "read_", "stream_", "option_", "stock_", "index_")
        )
    ]

    print(f"Discovered {len(public_methods)} facade helpers")
    for name in public_methods[:limit]:
        helper = getattr(theta, name)
        print(f"  - {name}{inspect.signature(helper)}")

    if len(public_methods) > limit:
        print(f"  ... {len(public_methods) - limit} more")


def test_facade_method_lookup(theta: ThetaDataRS) -> None:
    """Verify dynamic method lookup and error reporting without network calls."""
    for method_name in (
        "get_options_symbols_list",
        "get_options_contract_list",
        "option_quote_payload",
        "stream_stock_quotes",
    ):
        helper = getattr(theta, method_name)
        print(f"{method_name}: {inspect.signature(helper)}")

    try:
        theta.not_a_real_helper()
    except AttributeError as exc:
        print(f"missing helper check: {exc}")


def test_stream_payload_helpers(theta: ThetaDataRS) -> None:
    """Exercise streaming payload helpers; these are pure local functions."""
    quote_payload = theta.option_quote_payload(
        "AAPL",
        "2026-01-16",
        200,
        "C",
        request_id=101,
    )
    trade_payload = theta.stock_trade_payload("AAPL", request_id=102)
    index_payload = theta.index_price_payload("SPX", request_id=103)

    print("option_quote_payload:", quote_payload)
    print("stock_trade_payload:", trade_payload)
    print("index_price_payload:", index_payload)


def test_live_symbols(theta: ThetaDataRS, rows: int = 5) -> None:
    """Fetch a small option symbol sample through ThetaDataRS."""
    symbols = theta.get_options_symbols_list()
    print(f"get_options_symbols_list returned {len(symbols)} rows")
    print(symbols.head(rows))


def test_live_contracts(
    theta: ThetaDataRS,
    ticker: str = "AAPL",
    trading_date: dt.date | str | None = None,
    rows: int = 5,
) -> None:
    """Fetch a small option contract sample through ThetaDataRS."""
    if trading_date is None:
        trading_date = dt.date.today().isoformat()

    contracts = theta.get_options_contract_list(ticker, trading_date)
    print(
        f"get_options_contract_list({ticker!r}, {trading_date!r}) returned {len(contracts)} rows"
    )
    print(contracts.head(rows))


def run_smoke_tests(
    *,
    live: bool = False,
    ticker: str = "AAPL",
    trading_date: str | None = None,
    dataframe_return_type: str = "polars",
) -> None:
    theta = ThetaDataRS(dataframe_return_type=dataframe_return_type)

    show_available_helpers(theta)
    test_facade_method_lookup(theta)
    test_stream_payload_helpers(theta)

    if live:
        test_live_symbols(theta)
        test_live_contracts(theta, ticker=ticker, trading_date=trading_date)
    else:
        print("Skipping live ThetaData REST checks; pass --live to enable them.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the ThetaDataRS facade helpers."
    )
    parser.add_argument(
        "--live", action="store_true", help="Run small ThetaData REST calls."
    )
    parser.add_argument(
        "--ticker", default="AAPL", help="Ticker for the live contract-list check."
    )
    parser.add_argument(
        "--date",
        dest="trading_date",
        default=None,
        help="Trading date for the live contract-list check, formatted YYYY-MM-DD.",
    )
    parser.add_argument(
        "--dataframe-return-type",
        choices=("polars", "pandas"),
        default="polars",
        help="DataFrame type passed to the ThetaData client.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_smoke_tests(
        live=args.live,
        ticker=args.ticker,
        trading_date=args.trading_date,
        dataframe_return_type=args.dataframe_return_type,
    )


if __name__ == "__main__":
    main()
