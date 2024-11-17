import math

# 예금 계산 클래스
class Deposit:
    def __init__(self, rate: float):
        self.rate = rate / 100 / 12  # 월 이자율로 변환

    # 목표 금액 기준 예금 계산 (한 번에 예치할 금액 계산)
    def calculate_lump_sum(self, goal_amount: float, months: int):
        initial_amount = goal_amount / ((1 + self.rate) ** months)
        return initial_amount

# 적금 계산 클래스
class Saving:
    def __init__(self, rate: float):
        self.rate = rate / 100 / 12  # 월 이자율로 변환

    # 월 적립 기준 적금 계산
    def calculate_monthly_saving(self, monthly_saving: float, months: int):
        total = monthly_saving * ((1 + self.rate) ** months - 1) / self.rate
        return total

    # 목표 금액 기준 적금 계산 (목표 금액에 도달하기 위해 필요한 월 적립액 계산)
    def calculate_goal_amount(self, goal_amount: float, months: int):
        monthly_saving = goal_amount * self.rate / ((1 + self.rate) ** months - 1)
        return monthly_saving

# 대출 이자 계산 클래스
class LoanInterest:
    def __init__(self, principal: float, annual_rate: float, months: int):
        self.principal = principal
        self.rate = annual_rate / 100 / 12  # 월 이자율로 변환
        self.months = months

    # 대출 상환 계산 (월 상환액, 총 상환액, 이자 금액 계산)
    def calculate(self):
        monthly_payment = self.principal * self.rate * (1 + self.rate) ** self.months / ((1 + self.rate) ** self.months - 1)
        total_payment = monthly_payment * self.months
        interest_payment = total_payment - self.principal
        return {
            "monthly_payment": monthly_payment,
            "total_payment": total_payment,
            "interest_payment": interest_payment
        }



# 예시 사용법
if __name__ == "__main__":
    # 예금 계산 예시
    deposit = Deposit(rate=5)

    print(f"Final amount for lump sum: {deposit.calculate_lump_sum(2000000, 24):,.2f}원")

    # 적금 계산 예시
    saving = Saving(rate=5)
    print(f"Total saving (monthly basis): {saving.calculate_monthly_saving(100000, 24):,.2f}원")
    print(f"Monthly saving needed for goal: {saving.calculate_goal_amount(5000000, 24):,.2f}원")

    # 대출 이자 계산 예시
    loan = LoanInterest(principal=50000000, annual_rate=4.5, months=120)
    loan_result = loan.calculate()
    print(f"Monthly payment: {loan_result['monthly_payment']:,.2f}원")
    print(f"Total payment: {loan_result['total_payment']:,.2f}원")
    print(f"Total interest: {loan_result['interest_payment']:,.2f}원")






# tools_list 정의
tools_list = [
    {
        "name": "calculate_goal_amount_saving",
        "description": "목표 금액을 위해 매월 얼마를 저축해야 하는지 적금 방식으로 계산",
        "parameters": [
            {"name": "goal_amount", "type": "float", "description": "목표 금액"},
            {"name": "months", "type": "int", "description": "납입 기간 (개월)"},
            {"name": "rate", "type": "float", "description": "연 이자율 (%)"}
        ]
    },
    {
        "name": "calculate_lump_sum_deposit",
        "description": "목표 금액을 달성하기 위해 필요한 예치금을 예금 방식으로 계산",
        "parameters": [
            {"name": "goal_amount", "type": "float", "description": "목표 금액"},
            {"name": "months", "type": "int", "description": "기간 (개월)"},
            {"name": "rate", "type": "float", "description": "연 이자율 (%)"}
        ]
    },
    {
        "name": "calculate_monthly_saving",
        "description": "월 적립 기준 적금 총액 계산",
        "parameters": [
            {"name": "monthly_saving", "type": "float", "description": "월 적립 금액"},
            {"name": "months", "type": "int", "description": "납입 기간 (개월)"},
            {"name": "rate", "type": "float", "description": "연 이자율 (%)"}
        ]
    },
    {
        "name": "calculate_lump_sum",
        "description": "초기 예치금 기준으로 최종 예금 금액을 계산",
        "parameters": [
            {"name": "initial_amount", "type": "float", "description": "초기 예치금"},
            {"name": "months", "type": "int", "description": "기간 (개월)"},
            {"name": "rate", "type": "float", "description": "연 이자율 (%)"}
        ]
    },
    {
        "name": "calculate_loan_interest",
        "description": "대출 상환 계획에 따른 월 상환액, 총 상환액, 이자 금액 계산",
        "parameters": [
            {"name": "principal", "type": "float", "description": "대출 원금"},
            {"name": "annual_rate", "type": "float", "description": "연 이자율 (%)"},
            {"name": "months", "type": "int", "description": "상환 기간 (개월)"}
        ]
    }
]

# Function Calling 구현
def call_financial_function(function_name, **kwargs):
    if function_name == "calculate_goal_amount_saving":
        saving_calculator = Saving(rate=kwargs["rate"])
        return saving_calculator.calculate_goal_amount(goal_amount=kwargs["goal_amount"], months=kwargs["months"])
    
    elif function_name == "calculate_lump_sum_deposit":
        deposit_calculator = Deposit(rate=kwargs["rate"])
        return deposit_calculator.calculate_lump_sum(goal_amount=kwargs["goal_amount"], months=kwargs["months"])
    
    elif function_name == "calculate_monthly_saving":
        saving_calculator = Saving(rate=kwargs["rate"])
        return saving_calculator.calculate_monthly_saving(monthly_saving=kwargs["monthly_saving"], months=kwargs["months"])
    
    elif function_name == "calculate_lump_sum":
        deposit_calculator = Deposit(rate=kwargs["rate"])
        return deposit_calculator.calculate_lump_sum(initial_amount=kwargs["initial_amount"], months=kwargs["months"])
    
    elif function_name == "calculate_loan_interest":
        loan_calculator = LoanInterest(principal=kwargs["principal"], annual_rate=kwargs["annual_rate"], months=kwargs["months"])
        return loan_calculator.calculate()