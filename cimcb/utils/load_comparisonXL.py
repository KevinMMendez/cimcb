import pandas as pd


def load_comparisonXL(method, evaluate="train"):
    """Load comparison table."""
    # TO DO: Check that evaluate includes 'test'
    if evaluate is "test":
        e = "['Test']"
    elif evaluate is "in bag":
        e = "['In bag']"
    elif evaluate is "out of bag":
        e = "['Out of Bag']"
    else:
        e = "['Train']"

    # Import methods
    table = []
    for i in method:
        table.append(pd.read_excel(i + ".xlsx"))

    # Concatenate table
    df = pd.DataFrame()
    for i in range(len(table)):
        df = pd.concat([df, table[i].loc[table[i]['evaluate'] == e].T.squeeze()], axis=1, sort=False)
    df = df.T.drop(columns="evaluate")

    # Remove [ ] from string
    for i in range(len(df)):
        for j in range(len(df.T)):
            df.iloc[i, j] = df.iloc[i, j][2: -2]

    # Reset index and add methods column
    df = df.reset_index()
    df = pd.concat([pd.Series(method, name="method"), df], axis=1, sort=False)
    df = df.drop("index", 1)
    df = df.set_index("method")

    return df
