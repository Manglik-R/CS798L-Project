import random

def estimate_valid_k_grams(s1, sk, p):
    """
    s1: set of valid 1-grams
    sk: set of valid k-grams
    p: probability of sampling
    """

    N = max(1, int(p * len(s1) * len(sk)))
    count = 0
    if not s1 or not sk:
        return 0
    
    # print(f"Number of samples: {N}")

    for _ in range(N):
        x = random.choice(list(s1))
        w = random.choice(list(sk))
        len_w = len(w.split(" "))

        # if len_w < 2:
        #     y = w
        #     z = ""
        # else:
        #     y = " ".join(w.rsplit(" ", 1)[:-1])
        #     z = w.split(" ")[-1]

        y = " ".join(w.rsplit(" ", 1)[:-1])
        z = w.split(" ")[-1]


        xy = x if y == "" else f"{x} {y}"
        # print(f"x: -{x}-, y: -{y}-, sz: -{z}-, xy: -{xy}-")

        if (xy in sk) and (z in s1):
            # print(f"x: -{x}-, y: -{y}-, sz: -{z}-, xy: -{xy}-")
            count += 1

    return int(count / p)


def check_validity(w, s1, sk):
    """
    w: k+1 gram
    s1: set of valid 1-grams
    sk: set of valid k-grams
    """

    w_parts = w.split(" ")
    
    if len(w_parts) < 2:
        return False
    
    x = w_parts[0]
    y = " ".join(w_parts[1:-1])
    z = w_parts[-1]

    xy = x if y == "" else f"{x} {y}"
    yz = z if y == "" else f"{y} {z}"
    # print(f"x: -{x}-, y: -{y}-, z: -{z}-, xy: -{xy}-, yz: -{yz}-")
    if (x in s1) and (z in s1) and (xy in sk) and (yz in sk):
        return True
    else:
        return False

def prune_invalid(W, s1, sk):
    """
    W: list of k+1 grams
    s1: set of valid 1-grams
    sk: set of valid k-grams
    """

    W1 = [w for w in W if check_validity(w, s1, sk)]
    
    return set(W1)

def clean_text(text):
    """
    Removes all the non-albhanumeric characters from the text expect for spaces.
    """
    text = text.lower()
    return ''.join([c if c.isalnum() or c.isspace() else '' for c in text])

def prune_invalid_2grams(W, s1):
    """
    W: list of tuples of 2-grams of each user
    s1: set of valid 1-grams
    """

    for user_set in W:
        new_set = set()
        for w in user_set:
            x = w.split(" ")[0]
            y = w.split(" ")[1]
            if (x in s1) and (y in s1):
                new_set.add(w)
        W[W.index(user_set)] = new_set  
    return W