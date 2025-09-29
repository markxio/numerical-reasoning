import re
import pandas as pd

def main():
    df = pd.read_csv("FILTER.csv", sep=";")
    df = df.drop("passes_quality_filter", axis=1)    

    money = df[df["answer_labels"]=="['MONEY']"]
    money["has_dollar_sign_twice"] = money.apply(lambda row: has_sign_n_times(row.sentence, r"\$", 2), axis=1)
    money["has_dollar_sign_thrice"] = money.apply(lambda row: has_sign_n_times(row.sentence, r"\$", 3), axis=1)
    money["has_two_years"] = money.apply(lambda row: has_n_unique_consecutive_years(row.sentence, 2), axis=1)
    money["has_three_years"] = money.apply(lambda row: has_n_unique_consecutive_years(row.sentence, 3), axis=1)
    money.to_csv("FILTER_label_MONEY.csv", sep=";", index=False)

    percent = df[df["answer_labels"]=="['PERCENT']"]
    percent["has_percent_sign_twice"] = percent.apply(lambda row: has_sign_n_times(row.sentence, r"%", 2), axis=1)
    percent["has_percent_sign_thrice"] = percent.apply(lambda row: has_sign_n_times(row.sentence, r"%", 3), axis=1)
    percent["has_two_years"] = percent.apply(lambda row: has_n_unique_consecutive_years(row.sentence, 2), axis=1)
    percent["has_three_years"] = percent.apply(lambda row: has_n_unique_consecutive_years(row.sentence, 3), axis=1)
    percent.to_csv("FILTER_label_PERCENT.csv", sep=";", index=False)

def has_sign_n_times(mystr: str, sign: str, n: int):
    occurrences = re.findall(sign, mystr)
    if len(occurrences) == n:
        return True
    return False

def has_n_unique_consecutive_years(mystr: str, n: int):
    occurrences = re.findall(r"\d{4}", mystr)
    # unique digits
    # e.g., avoid the same year mentioned twice
    if len(list(set(occurrences))) != n:
        return False

    # sorts descending
    sorted_years = sorted(occurrences, reverse=True)
    print(sorted_years) 
    for idx, digits in enumerate(sorted_years):
        year = int(digits)
        # ensure the four digits could be dates
        # within the reporting horizon
        if year < 1993 or year > 2024: 
            return False
    
        if idx > 0:
            prev_year=int(sorted_years[idx-1]) 
            if 1 != prev_year - year:
                # not consecutive
                return False
    print(True)
    return True

if __name__=="__main__":
    main()
