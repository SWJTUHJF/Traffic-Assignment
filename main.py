from algorithm import solve


def main():
    solve(network_name="SiouxFalls",
          TA_type="UE",
          main_algorithm="FW",
          spp_algorithm="LS",
          bisection_gap=1e-4,
          main_gap=1e-4,
          verbose=True)


if __name__ == '__main__':
    main()

