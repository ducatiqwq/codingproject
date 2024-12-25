user_prompt = """User:
Calculate {number_1}+{number_2} in vertical form, and write down the detailed steps. Put your final answer within \\boxed{{}}.
"""

model_response = """Assistant:
To calculate $ {number_1} + {number_2} $ in vertical form, follow these steps:

1. **Align the Numbers Vertically**: Write the numbers so that each digit is in its respective column, aligning the units, tens, hundreds, etc.

   ```
     {number_1:>5}
   + {number_2:>5}
   ```

2. **Add Starting from the Rightmost Column**: Begin adding from the rightmost column (units place) and move left.

   ```
     {number_1:>5}
   + {number_2:>5}
   -------
   ```

3. **Add Each Column**: {colums_addition}

4. **Write Down the Result**:

   ```
     {number_1:>5}
   + {number_2:>5}
   -------
     {numbers_sum:>5}
   ```

5. **Final Answer**: The final sum is $ \\boxed{{{numbers_sum}}} $.
"""

if __name__ == "__main__":
	print(user_prompt.format(number_1=7823, number_2=6939))