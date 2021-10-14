if __name__ == '__main__' and __package__ is None:

    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from dotenv import load_dotenv, find_dotenv

    # only to allow imports outside current folder
    print("running notebook configuration")
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    root_path = os.environ.get("LOCAL_PATH")
    os.sys.path.append(root_path)

    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 50)
    pd.set_option('max_info_columns', 5)
    pd.set_option('precision', 4)
    pd.options.display.float_format = '{:,.4f}'.format

    plt.rcParams.update({'font.size': 16,
                     'figure.figsize': (12, 6),
                     'lines.linewidth': 4})
    # sns.set_context("paper")

