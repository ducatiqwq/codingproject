from typing import Tuple

column_descriptions = [
	"Units",
	"Tens",
	"Hundreds",
	"Thousands",
	"Ten Thousands",
	"Hundred Thousands",
	"Millions",
	"Ten Millions",
	"Hundred Millions"
]

def construct_column_addition(x_dig: int, y_dig: int, carry: int, column_description: str) -> Tuple[str, int]:
	result = x_dig + y_dig + carry
	if carry:
		calculation = f"   - **{column_description} Column**: $ {x_dig} + {y_dig} + {carry} = {result} $ "
		if result >= 10:
			description = f"(write down {result % 10}, carry over {result // 10})"
		else:
			description = "(carry from previous)"
	
	else:
		calculation = f"   - **{column_description} Column**: $ {x_dig} + {y_dig} = {result} $ "
		if result >= 10:
			description = f"(write down {result % 10}, carry over {result // 10})"
		else:
			description = ""

	return calculation + description, result // 10


def construct_colums_addition(x: int, y: int):
	carry = 0
	additions = ""

	for i in range(100):
		x_dig, x = x % 10, x // 10
		y_dig, y = y % 10, y // 10
		if x_dig == 0 and y_dig == 0 and carry == 0:
			break

		addition, new_carry = construct_column_addition(x_dig, y_dig, carry, column_descriptions[i])
		additions += '\n' + addition
		carry = new_carry

	return additions