import json
import os
import random

BASE_QUERIES = [
    "What is the company refund policy?",
    "Tell me about the engineering team.",
    "What is the remote work policy?",
    "Explain the vacation policy.",
    "How do I reset my password?",
    "What is the office dress code?",
    "How do I submit an expense report?",
    "What are the working hours?",
    "Is parking free at the office?",
    "Who is the head of HR?",
    "What is the company policy on travel expenses?",
    "How do I log my hours in the timesheet?",
    "Is coffee free at the cafeteria?",
    "What is the maternity leave policy?",
    "How do I refer a candidate for a job opening?",
    "Where is the headquarters of the company?",
    "What are the company core values?",
    "How do I register for health insurance?",
    "When is the next company all-hands meeting?",
    "What are the security procedures for visitors?",
    "How do I request a new laptop?",
    "What is the performance review process?",
    "Are there any employee discounts available?",
    "How do I book a meeting room?",
    "What is the policy on social media usage?",
]

SEMANTIC_VARIANTS = {
    "What is the company refund policy?": [
        "How do I get a refund?",
        "Can you explain the refund process?",
        "What are the rules for product refunds?",
        "How are refunds handled here?",
        "Where can I read about refunds?",
    ],
    "Tell me about the engineering team.": [
        "Who works in engineering?",
        "What does the engineering department do?",
        "Can you describe the engineering team structure?",
        "How is the engineering organization set up?",
        "Who is on the dev team?",
    ],
    "What is the remote work policy?": [
        "Can I work from home?",
        "What are the rules for WFH?",
        "Explain our remote working guidelines.",
        "Is remote work allowed?",
        "How does the work from home policy function?",
    ],
    "Explain the vacation policy.": [
        "How many PTO days do I get?",
        "What is our paid time off policy?",
        "How do I request vacation time?",
        "Explain the rules for taking time off.",
        "What is the company policy on vacations?",
    ],
    "How do I reset my password?": [
        "I forgot my password, how to recover it?",
        "Where can I change my account password?",
        "What is the procedure for a password reset?",
        "How do I unlock my account with a new password?",
        "Reset password instructions please.",
    ],
    "What is the office dress code?": [
        "What are we allowed to wear at work?",
        "Is there a dress code policy?",
        "Can I wear casual clothes to the office?",
        "What is the employee dress code guideline?",
        "Are jeans allowed in the office?",
    ],
    "How do I submit an expense report?": [
        "Where do I file my expenses?",
        "What is the process to claim business expenses?",
        "How do I get reimbursed for travel?",
        "Submit expense report steps.",
        "How do I report my monthly expenses?",
    ],
    "What are the working hours?": [
        "When do I need to be in the office?",
        "What is the standard work schedule?",
        "What are our core office hours?",
        "When does the workday start and end?",
        "What hours are employees expected to work?",
    ],
    "Is parking free at the office?": [
        "Do I have to pay to park my car at work?",
        "What is the parking situation at the headquarters?",
        "Are there free parking spaces for employees?",
        "How does office parking work?",
        "Do we get free parking passes?",
    ],
    "Who is the head of HR?": [
        "Who leads the Human Resources department?",
        "Who is in charge of HR?",
        "Can you tell me the name of the HR director?",
        "Who runs our HR team?",
        "Who is the chief people officer?",
    ],
    "What is the company policy on travel expenses?": [
        "How are travel costs reimbursed?",
        "What travel expenses are covered by the company?",
        "Explain the business travel policy.",
        "What is the limit for travel expense claims?",
        "How do I request travel budget?",
    ],
    "How do I log my hours in the timesheet?": [
        "Where do I enter my weekly work hours?",
        "What is the timesheet submission process?",
        "How do I submit my hours?",
        "Instructions for loging hours.",
        "How do I fill out my timesheet?",
    ],
    "Is coffee free at the cafeteria?": [
        "Do we get free coffee at work?",
        "Are drinks free in the cafeteria?",
        "Is there free coffee for employees?",
        "Do we have to pay for coffee?",
        "What beverages are free at the office?",
    ],
    "What is the maternity leave policy?": [
        "How long is maternity leave?",
        "Explain the parental leave guidelines.",
        "What is the company policy on pregnancy leave?",
        "How do I apply for maternity leave?",
        "Is maternity leave paid?",
    ],
    "How do I refer a candidate for a job opening?": [
        "What is the employee referral process?",
        "How do I submit a candidate referral?",
        "Where can I refer someone for a job?",
        "Explain the job referral program.",
        "How do I recommend a friend for a role?",
    ],
    "Where is the headquarters of the company?": [
        "Where is our main office located?",
        "What is the address of the company headquarters?",
        "Where is the corporate office?",
        "Where are we headquartered?",
        "HQ location address.",
    ],
    "What are the company core values?": [
        "What does the company stand for?",
        "Explain our organizational values.",
        "What are our cultural pillars?",
        "Describe the company's core values.",
        "What is the company mission and values?",
    ],
    "How do I register for health insurance?": [
        "What is the health insurance signup process?",
        "How do I enroll in the medical plan?",
        "Where can I select my health benefits?",
        "Enroll in health insurance guide.",
        "How do I add dependents to my health plan?",
    ],
    "When is the next company all-hands meeting?": [
        "What date is the next all-hands?",
        "When is the employee town hall meeting?",
        "Schedule for the next company-wide meeting.",
        "When do we have the next all-hands?",
        "Is there a town hall scheduled soon?",
    ],
    "What are the security procedures for visitors?": [
        "How do guests sign in at the office?",
        "What are the visitor guidelines?",
        "What is the process for visitor entry?",
        "How do I register a guest?",
        "Visitor badge policy.",
    ],
    "How do I request a new laptop?": [
        "What is the process for hardware upgrades?",
        "How do I get a replacement computer?",
        "Request new IT equipment process.",
        "My laptop is slow, how do I get a new one?",
        "IT hardware provisioning policy.",
    ],
    "What is the performance review process?": [
        "How are annual reviews conducted?",
        "Explain the performance evaluation guidelines.",
        "When do performance reviews occur?",
        "How do self-evaluations work here?",
        "Describe the feedback cycle process.",
    ],
    "Are there any employee discounts available?": [
        "Do we get corporate discounts?",
        "What benefits or perks offer employee discounts?",
        "Are there partnerships with discounts for staff?",
        "How do I access employee perks and discounts?",
        "List of employee corporate discounts.",
    ],
    "How do I book a meeting room?": [
        "Where can I reserve a conference room?",
        "How do I schedule a meeting room in Outlook?",
        "What is the room booking process?",
        "Reserve a meeting space instructions.",
        "How do I check conference room availability?",
    ],
    "What is the policy on social media usage?": [
        "What are the social media guidelines for employees?",
        "Is there a company policy on posting online?",
        "Are employees allowed to talk about work on Twitter?",
        "Describe our internet and social media policy.",
        "What are the rules for posting on personal social media?",
    ],
}

NOVEL_QUERIES = [
    "What is the distance between the Earth and the Moon?",
    "Who wrote the play Hamlet?",
    "What is the chemical symbol for gold?",
    "How many continents are there on Earth?",
    "What is the capital of Japan?",
    "Who was the first president of the United States?",
    "What is the speed of light in a vacuum?",
    "How does photosynthesis work?",
    "What is the tallest mountain in the world?",
    "Who painted the Mona Lisa?",
    "What is the largest ocean on Earth?",
    "What is the atomic number of carbon?",
    "Who discovered gravity?",
    "What is the primary function of red blood cells?",
    "How many bones are in the adult human body?",
    "What is the longest river in the world?",
    "Who developed the theory of relativity?",
    "What is the capital of Australia?",
    "What is the boiling point of water in Celsius?",
    "Who was the first man to step on the Moon?",
    "What is the largest country in the world by land area?",
    "How do computers store data in binary?",
    "What is the population of the United Kingdom?",
    "Who is the current Prime Minister of Canada?",
    "What is the currency of Brazil?",
    "What causes earthquakes to occur?",
    "Who wrote the book 1984?",
    "What is the smallest country in the world?",
    "How does an electric motor work?",
    "What is the capital of Canada?",
    "How many states are in the United States?",
    "What is the largest planet in our solar system?",
    "Who is the author of To Kill a Mockingbird?",
    "What is the currency of India?",
    "What are the primary colors in painting?",
    "Who was Albert Einstein?",
    "What is the main language spoken in Mexico?",
    "How do airplanes fly?",
    "What is the capital of Germany?",
    "How many teeth does an adult human have?",
    "Who invented the telephone?",
    "What is the capital of Italy?",
    "What is the square root of 144?",
    "How many days are in a leap year?",
    "What is the national animal of India?",
    "Who wrote Pride and Prejudice?",
    "What is the capital of Spain?",
    "What is the largest desert in the world?",
    "How does a database index speed up queries?",
    "What is the difference between TCP and UDP?",
    "What is HTTP status code 404?",
    "Who founded the company Microsoft?",
    "What is the capital of France?",
    "Who is the CEO of Apple?",
    "What is the national flag of Canada?",
    "How many players are on a soccer field?",
    "Who wrote Romeo and Juliet?",
    "What is the capital of Russia?",
    "What is the currency of South Africa?",
    "How do batteries store electrical energy?",
    "What is the largest land mammal?",
    "Who was Marie Curie?",
    "What is the capital of China?",
    "What is the definition of a prime number?",
    "How many hours are in a week?",
    "Who wrote the Odyssey?",
    "What is the capital of India?",
    "What is the currency of Mexico?",
    "What is the primary crop grown in Iowa?",
    "How does a barcode scanner work?",
    "What is the tallest building in New York?",
    "Who invented the light bulb?",
    "What is the capital of Egypt?",
    "What is the definition of a CPU?",
    "How many minutes are in a day?",
    "Who is the author of Harry Potter?",
    "What is the capital of Greece?",
    "What is the currency of Japan?",
    "What is the main ingredient in bread?",
    "How does GPS tracking function?",
    "What is the largest lake in North America?",
    "Who was Isaac Newton?",
    "What is the capital of Argentina?",
    "What is the definition of mitosis?",
    "How many seconds are in an hour?",
    "Who wrote the Iliad?",
    "What is the capital of Sweden?",
    "What is the currency of the United Kingdom?",
    "What is the main element in the Sun?",
    "How does an engine cooling system work?",
    "What is the tallest tree species in the world?",
    "Who invented the printing press?",
    "What is the capital of Mexico?",
    "What is the definition of RAM?",
    "How many weeks are in a year?",
    "Who is the author of The Great Gatsby?",
    "What is the capital of Norway?",
    "What is the currency of South Korea?",
    "What is the main ingredient in chocolate?",
    "How does a touch screen detect inputs?",
    "What is the largest island in the world?",
    "Who was Charles Darwin?",
    "What is the capital of Turkey?",
    "What is the definition of photosynthesis?",
    "How many months have 31 days?",
    "Who wrote The Catcher in the Rye?",
    "What is the capital of Denmark?",
    "What is the currency of Russia?",
    "What is the main element in water?",
    "How does a refrigerator keep food cold?",
    "What is the fastest animal on land?",
    "Who invented the steam engine?",
    "What is the capital of Colombia?",
    "What is the definition of a glacier?",
    "How many degrees are in a circle?",
    "Who wrote The Lord of the Rings?",
    "What is the capital of Portugal?",
    "What is the currency of Switzerland?",
    "What is the main crop of Kansas?",
    "How does a solar panel generate electricity?",
]


def generate_workload():
    random.seed(42)  # Maintain deterministic splits for repeatable benchmarking

    total_queries = 1000
    exact_count = int(total_queries * 0.30)  # 300 exact repeats
    semantic_count = int(total_queries * 0.40)  # 400 semantic variations
    novel_count = int(total_queries * 0.30)  # 300 novel queries

    workload = []

    # 1. Generate Exact (30%) - pick from base queries
    for _ in range(exact_count):
        query = random.choice(BASE_QUERIES)
        workload.append({"query": query, "type": "exact"})

    # 2. Generate Semantic (40%) - pick from semantic variants
    for _ in range(semantic_count):
        # Pick a base query that has variants
        base = random.choice(list(SEMANTIC_VARIANTS.keys()))
        variant = random.choice(SEMANTIC_VARIANTS[base])
        workload.append({"query": variant, "type": "semantic"})

    # 3. Generate Novel (30%) - pick from novel queries
    # If we need 300 novel queries and only have ~120 in the pool,
    # we will select with replacements.
    for _ in range(novel_count):
        query = random.choice(NOVEL_QUERIES)
        workload.append({"query": query, "type": "novel"})

    # Shuffle the workload to represent randomized production scheduling
    random.shuffle(workload)

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "benchmark_workload.json")
    with open(output_path, "w") as f:
        json.dump(workload, f, indent=2)

    print(f"Generated {len(workload)} queries:")
    print(f"  - Exact  : {exact_count} (30%)")
    print(f"  - Semantic: {semantic_count} (40%)")
    print(f"  - Novel   : {novel_count} (30%)")
    print(f"Successfully saved to {output_path}")


if __name__ == "__main__":
    generate_workload()
