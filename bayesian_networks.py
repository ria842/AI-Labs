import random


# ============================================================
# 1. TRAINING DATA
# ============================================================

sentences = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug"
]


# Add <START> and <END> to every sentence
training_data = []

for sentence in sentences:
    tokens = sentence.split()

    tokens = ["<START>"] + tokens + ["<END>"]

    training_data.append(tokens)


# ============================================================
# 2. FIRST-ORDER MODEL
# ============================================================

# First-order model:
#
# P(X_t | X_(t-1))
#
# transition_counts_1[current_token][next_token]
#
# Example:
#
# transition_counts_1["cat"]["sat"] = 2

transition_counts_1 = {}


for sentence in training_data:

    for i in range(len(sentence) - 1):

        current_token = sentence[i]
        next_token = sentence[i + 1]

        if current_token not in transition_counts_1:
            transition_counts_1[current_token] = {}

        if next_token not in transition_counts_1[current_token]:
            transition_counts_1[current_token][next_token] = 0

        transition_counts_1[current_token][next_token] += 1


# ============================================================
# 3. FIRST-ORDER CONDITIONAL PROBABILITIES
# ============================================================

# P(next_token | current_token)

transition_probs_1 = {}


for current_token in transition_counts_1:

    transition_probs_1[current_token] = {}

    total_count = sum(
        transition_counts_1[current_token].values()
    )

    for next_token in transition_counts_1[current_token]:

        count = transition_counts_1[current_token][next_token]

        probability = count / total_count

        transition_probs_1[current_token][next_token] = probability


# ============================================================
# 4. SECOND-ORDER MODEL
# ============================================================

# Second-order model:
#
# P(X_t | X_(t-2), X_(t-1))
#
# The key is a pair of previous tokens:
#
# (previous_previous_token, previous_token)
#
# Example:
#
# ("the", "cat") -> "sat"
#
# ("cat", "sat") -> "on"
#
# second_order_counts[("the", "cat")]["sat"] = 2

second_order_counts = {}


for sentence in training_data:

    # We need at least TWO previous tokens.
    #
    # For:
    #
    # <START> the cat sat ...
    #
    # the first prediction is:
    #
    # P(cat | <START>, the)

    for i in range(2, len(sentence)):

        previous_previous_token = sentence[i - 2]
        previous_token = sentence[i - 1]
        next_token = sentence[i]

        context = (
            previous_previous_token,
            previous_token
        )

        if context not in second_order_counts:
            second_order_counts[context] = {}

        if next_token not in second_order_counts[context]:
            second_order_counts[context][next_token] = 0

        second_order_counts[context][next_token] += 1


# ============================================================
# 5. SECOND-ORDER CONDITIONAL PROBABILITIES
# ============================================================

# P(next_token | previous_previous_token, previous_token)

second_order_probs = {}


for context in second_order_counts:

    second_order_probs[context] = {}

    total_count = sum(
        second_order_counts[context].values()
    )

    for next_token in second_order_counts[context]:

        count = second_order_counts[context][next_token]

        probability = count / total_count

        second_order_probs[context][next_token] = probability


# ============================================================
# 6. DISPLAY FIRST-ORDER DISTRIBUTION
# ============================================================

def display_first_order_distribution(current_token):
    """
    Display:

        P(next_token | current_token)
    """

    print(f"\nP(next_token | '{current_token}')")

    if current_token not in transition_probs_1:
        print("No transitions found.")
        return

    for next_token, probability in transition_probs_1[current_token].items():
        print(f"  {next_token:10s} : {probability:.4f}")


# ============================================================
# 7. DISPLAY SECOND-ORDER DISTRIBUTION
# ============================================================

def display_second_order_distribution(
    previous_previous_token,
    previous_token
):
    """
    Display:

        P(next_token |
          previous_previous_token,
          previous_token)
    """

    context = (
        previous_previous_token,
        previous_token
    )

    print(
        f"\nP(next_token | '{previous_previous_token}', "
        f"'{previous_token}')"
    )

    if context not in second_order_probs:
        print("No transitions found.")
        return

    for next_token, probability in second_order_probs[context].items():
        print(f"  {next_token:10s} : {probability:.4f}")


# ============================================================
# 8. FIRST-ORDER GREEDY PREDICTION
# ============================================================

def predict_next_token_1(current_token):
    """
    First-order greedy prediction:

        argmax_w P(w | current_token)
    """

    if current_token not in transition_probs_1:
        return None

    next_token = max(
        transition_probs_1[current_token],
        key=transition_probs_1[current_token].get
    )

    return next_token


# ============================================================
# 9. FIRST-ORDER SAMPLING
# ============================================================

def sample_next_token_1(current_token):
    """
    First-order probabilistic sampling:

        P(next_token | current_token)
    """

    if current_token not in transition_probs_1:
        return None

    tokens = list(
        transition_probs_1[current_token].keys()
    )

    probabilities = list(
        transition_probs_1[current_token].values()
    )

    return random.choices(
        tokens,
        weights=probabilities,
        k=1
    )[0]


# ============================================================
# 10. SECOND-ORDER GREEDY PREDICTION
# ============================================================

def predict_next_token_2(
    previous_previous_token,
    previous_token
):
    """
    Second-order greedy prediction:

        argmax_w
        P(w | previous_previous_token, previous_token)
    """

    context = (
        previous_previous_token,
        previous_token
    )

    if context not in second_order_probs:
        return None

    next_token = max(
        second_order_probs[context],
        key=second_order_probs[context].get
    )

    return next_token


# ============================================================
# 11. SECOND-ORDER SAMPLING
# ============================================================

def sample_next_token_2(
    previous_previous_token,
    previous_token
):
    """
    Second-order probabilistic sampling:

        P(next_token |
          previous_previous_token,
          previous_token)
    """

    context = (
        previous_previous_token,
        previous_token
    )

    if context not in second_order_probs:
        return None

    tokens = list(
        second_order_probs[context].keys()
    )

    probabilities = list(
        second_order_probs[context].values()
    )

    return random.choices(
        tokens,
        weights=probabilities,
        k=1
    )[0]


# ============================================================
# 12. FIRST-ORDER SENTENCE GENERATION
# ============================================================

def generate_sentence_1(
    mode="sampling",
    max_steps=20
):
    """
    Generate using the first-order model:

        P(X_t | X_(t-1))
    """

    current_token = "<START>"
    generated_tokens = []

    for step in range(max_steps):

        if mode == "greedy":

            next_token = predict_next_token_1(
                current_token
            )

        elif mode == "sampling":

            next_token = sample_next_token_1(
                current_token
            )

        else:

            raise ValueError(
                "mode must be 'greedy' or 'sampling'"
            )

        if next_token is None:
            break

        if next_token == "<END>":
            break

        generated_tokens.append(next_token)

        current_token = next_token

    return " ".join(generated_tokens)


# ============================================================
# 13. SECOND-ORDER SENTENCE GENERATION
# ============================================================

def generate_sentence_2(
    mode="sampling",
    max_steps=20
):
    """
    Generate using the second-order model:

        P(X_t | X_(t-2), X_(t-1))

    The first token is generated from:

        P(X_1 | <START>)

    Then the second-order model takes over:

        P(X_2 | <START>, X_1)
        P(X_3 | X_1, X_2)
        P(X_4 | X_2, X_3)
        ...
    """

    generated_tokens = []

    # --------------------------------------------------------
    # Generate the first token.
    #
    # We only have one <START> in the training data, so
    # the first token must be obtained from the first-order
    # <START> distribution.
    # --------------------------------------------------------

    if mode == "greedy":

        first_token = predict_next_token_1("<START>")

    elif mode == "sampling":

        first_token = sample_next_token_1("<START>")

    else:

        raise ValueError(
            "mode must be 'greedy' or 'sampling'"
        )

    if first_token is None or first_token == "<END>":
        return ""

    generated_tokens.append(first_token)

    # --------------------------------------------------------
    # Now we have:
    #
    # <START>, first_token
    #
    # which gives us the two-token context required by
    # the second-order model.
    # --------------------------------------------------------

    previous_previous_token = "<START>"
    previous_token = first_token

    for step in range(max_steps - 1):

        # ----------------------------------------------------
        # GREEDY
        # ----------------------------------------------------

        if mode == "greedy":

            next_token = predict_next_token_2(
                previous_previous_token,
                previous_token
            )

        # ----------------------------------------------------
        # SAMPLING
        # ----------------------------------------------------

        elif mode == "sampling":

            next_token = sample_next_token_2(
                previous_previous_token,
                previous_token
            )

        # ----------------------------------------------------
        # No transition
        # ----------------------------------------------------

        else:
            raise ValueError(
                "mode must be 'greedy' or 'sampling'"
            )

        if next_token is None:
            break

        # Stop when <END> is generated
        if next_token == "<END>":
            break

        generated_tokens.append(next_token)

        # Shift the two-token context
        previous_previous_token = previous_token
        previous_token = next_token

    return " ".join(generated_tokens)


# ============================================================
# 14. DISPLAY FIRST-ORDER PROBABILITIES
# ============================================================

print("FIRST-ORDER TRANSITION PROBABILITIES")
print("=====================================")

for token in ["the", "cat", "dog", "sat", "ran"]:

    display_first_order_distribution(token)


# ============================================================
# 15. DISPLAY SECOND-ORDER PROBABILITIES
# ============================================================

print("\nSECOND-ORDER TRANSITION PROBABILITIES")
print("======================================")

contexts_to_display = [
    ("<START>", "the"),
    ("the", "cat"),
    ("the", "dog"),
    ("cat", "sat"),
    ("dog", "sat"),
    ("cat", "ran"),
    ("dog", "ran"),
    ("sat", "on"),
    ("ran", "to")
]

for previous_previous_token, previous_token in contexts_to_display:

    display_second_order_distribution(
        previous_previous_token,
        previous_token
    )


# ============================================================
# 16. FIRST-ORDER GREEDY GENERATION
# ============================================================

print("\nFIRST-ORDER GREEDY GENERATION")
print("=============================")

for i in range(5):

    sentence = generate_sentence_1(
        mode="greedy",
        max_steps=20
    )

    print(f"{i + 1}. {sentence}")


# ============================================================
# 17. FIRST-ORDER SAMPLING GENERATION
# ============================================================

print("\nFIRST-ORDER SAMPLING GENERATION")
print("===============================")

for i in range(5):

    sentence = generate_sentence_1(
        mode="sampling",
        max_steps=20
    )

    print(f"{i + 1}. {sentence}")


# ============================================================
# 18. SECOND-ORDER GREEDY GENERATION
# ============================================================

print("\nSECOND-ORDER GREEDY GENERATION")
print("==============================")

for i in range(5):

    sentence = generate_sentence_2(
        mode="greedy",
        max_steps=20
    )

    print(f"{i + 1}. {sentence}")


# ============================================================
# 19. SECOND-ORDER SAMPLING GENERATION
# ============================================================

print("\nSECOND-ORDER SAMPLING GENERATION")
print("================================")

for i in range(5):

    sentence = generate_sentence_2(
        mode="sampling",
        max_steps=20
    )

    print(f"{i + 1}. {sentence}")