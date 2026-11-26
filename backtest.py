          f"Okres testowy OOS: {test_start.date()} - "
        f"{test_end.date()}"
    )

    holding_comparison = {}

    for holding in HOLDING_PERIODS:

        print(
            f"\nBACKTEST: testowanie trzymania "
            f"pozycji przez {holding} sesji..."
        )

        train_result = run_portfolio_simulation(
            market,
            calendar,
            train_start,
            train_end,
            holding
        )

        test_result = run_portfolio_simulation(
            market,
            calendar,
            test_start,
            test_end,
            holding
        )

        holding_comparison[holding] = {
            "train": train_result,
            "out_of_sample": test_result
        }

    selected = holding_comparison[
        DEFAULT_HOLDING_SESSIONS
    ]

    benchmark = benchmark_return(
        test_start,
        test_end
    )

    report = {
        "version": VERSION,
        "instruments": len(market),
        "calendar_sessions": len(calendar),
        "train": selected["train"],
        "out_of_sample": selected["out_of_sample"],
        "benchmark": benchmark,
        "benchmark_name": "WIG",
        "test_start": test_start,
        "test_end": test_end,
        "costs_included": True,
        "commission_rate": COMMISSION_RATE,
        "slippage_rate": SLIPPAGE_RATE,
        "holding_sessions": DEFAULT_HOLDING_SESSIONS,
        "holding_comparison": holding_comparison
    }

    # ========================================================
    # RAPORT KONSOLI
    # ========================================================

    print("\n" + "=" * 65)
    print(f"WYNIKI BACKTESTU MARKET AI v{VERSION}")
    print("=" * 65)

    for holding in HOLDING_PERIODS:

        result = holding_comparison[holding]["out_of_sample"]

        print(f"\nTRZYMANIE: {holding} SESJI")
        print(
            f"Kapitał końcowy: "
            f"{result['ending_capital']:,.2f} PLN"
        )
        print(f"Wynik netto: {result['return']:+.2f}%")
        print(f"Wynik brutto: {result['gross_return']:+.2f}%")
        print(
            f"Maksymalne obsunięcie: "
            f"{result['drawdown']:.2f}%"
        )
        print(f"Liczba transakcji: {result['trades']}")
        print(f"Skuteczność: {result['win_rate']:.2f}%")
        print(
            f"Średni wynik transakcji: "
            f"{result['average_trade']:+.2f} PLN"
        )
        print(
            f"Prowizje łącznie: "
            f"{result['total_commission']:,.2f} PLN"
        )
        print(
            f"Poślizg łącznie: "
            f"{result['total_slippage']:,.2f} PLN"
        )
        print(
            f"Wszystkie koszty: "
            f"{result['total_transaction_costs']:,.2f} PLN"
        )
        print(
            f"Kontrola księgowa: "
            f"{result['reconciliation_error']:.8f} PLN"
        )

        pf = result["profit_factor"]

        if math.isinf(pf):
            pf_text = "brak stratnych transakcji"
        else:
            pf_text = f"{pf:.2f}"

        print(f"Profit factor: {pf_text}")

    print("\n" + "=" * 65)
    print("PORÓWNANIE Z WIG")
    print("=" * 65)

    if benchmark is None:
        print("Brak danych benchmarku WIG.")
    else:
        print(f"WIG: {benchmark:+.2f}%")

    print(
        f"MARKET AI ({DEFAULT_HOLDING_SESSIONS} sesje): "
        f"{selected['out_of_sample']['return']:+.2f}%"
    )
