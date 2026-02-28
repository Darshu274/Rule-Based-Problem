import argparse
import numpy as np
import pandas as pd
import operator 

def framework(pairs, arr):
    """
    Args:
       - pairs:  a list of (cond, calc) tuples. calc() must be an executable
       - arr: a numpy array with the features in order feat_1, feat_2, ...
    
    Executes the first calc() whose cond returns True.
    Returns None if no condition matches.
    """
    targets = []

    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
        
    return targets


def cond_eval(condition, arr):
    """evaluate a condition
        - condition: must be a tupe of (int, string, float). The second entry must be a string from the list below, describing the operator. Third entry of the tuple must be a float). If condition is None, it is always evaluated to true.
        - arr: array on which the condition is evaluated

    The python operator package is used. Second entry in condition must be one of those:
       ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    """
    ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition is None:
        return True
    
    op = ops[condition[1]]
    return op(arr[condition[0]], condition[2])


def main(args):

    # -----------------------------
    # Feature indices (from CSV)
    # -----------------------------
    F203 = 203
    F74  = 74

    # -----------------------------
    # Calculation functions
    # -----------------------------
    def calc_0_07(arr):  return 0.07
    def calc_0_96(arr):  return 0.96
    def calc_2_05(arr):  return 2.05
    def calc_2_92(arr):  return 2.92
    def calc_m1_17(arr): return -1.17
    def calc_m0_33(arr): return -0.33
    def calc_m0_72(arr): return -0.72

    # -----------------------------
    # Wrapper calc functions
    # (handle second condition)
    # -----------------------------
    def rule_1(arr):
        if arr[F74] <= 0.50:
            return calc_0_07(arr)
        return None

    def rule_2(arr):
        if arr[F74] > 0.50:
            return calc_0_96(arr)
        return None

    def rule_3(arr):
        if arr[F74] <= 0.43:
            return calc_2_05(arr)
        return None

    def rule_4(arr):
        if arr[F74] > 0.43:
            return calc_2_92(arr)
        return None

    def rule_5(arr):
        if arr[F74] <= 0.51:
            return calc_m1_17(arr)
        return None

    def rule_6(arr):
        if arr[F74] > 0.51:
            return calc_m0_33(arr)
        return None

    def rule_7(arr):
        if arr[F74] <= 0.51:
            return calc_m0_33(arr)
        return None

    def rule_8(arr):
        return calc_m0_72(arr)

    # -----------------------------
    # Condition–calculation pairs
    # (FLAT, ORDERED, EXPLICIT)
    # -----------------------------
    pair_list = [
        ((F203, "<=", 0.20), rule_1),
        ((F203, "<=", 0.20), rule_2),

        ((F203, "<=", 0.50), rule_3),
        ((F203, "<=", 0.50), rule_4),

        ((F203, "<=", 0.70), rule_5),
        ((F203, "<=", 0.70), rule_6),

        ((F203, ">",  0.70), rule_7),
        (None,              rule_8)   # fallback
    ]

    # -----------------------------
    # Load eval data & predict
    # -----------------------------
    data_array = pd.read_csv(args.eval_file_path).values

    return framework(pair_list, data_array)

    
def main_example(args):

    # Example: 
    test_arr = np.ones((10,10))

    def calc1(arr):
        """square first array column"""
        return arr[0]**2

    def calc2(arr):
        """add columns 3 and 4"""
        return arr[2] + arr[3]

    condition1 = (0,">=", 0.5)
    condition2 = (8, "==", 0.0)

    predict_targets = framework([(condition1, calc1), (condition2, calc2)], test_arr)
    print (predict_targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Task 2")
    parser.add_argument("--eval_file_path", required=True, help="Path to EVAL_58.csv")
    args = parser.parse_args()

    target02 = main(args)
