# # ============================================
# # MULTILAYER MODEL (WHY HIDDEN LAYERS MATTER)
# # Goal: Learn the XOR problem
# #
# # XOR rule (the "exclusive or"):
# #   0 , 0  -> 0   (same    -> 0)
# #   0 , 1  -> 1   (different-> 1)
# #   1 , 0  -> 1   (different-> 1)
# #   1 , 1  -> 0   (same    -> 0)
# #
# # THE BIG IDEA:
# #   A single linear layer draws ONE straight line to split the data.
# #   But XOR cannot be split by any single straight line.
# #   So the linear model is STUCK and fails.
# #   Add ONE hidden layer (+ a non-linear activation) and it JUST WORKS.
# # ============================================

import torch                # Main PyTorch library
import torch.nn as nn       # Layers, models, loss functions
import torch.optim as optim # Optimizers (update the weights)


# ============================================
# STEP 1: THE DATA (the 4 XOR cases)
# ============================================

# X = inputs -> 4 samples, 2 features each
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])

# y = correct answers for XOR
y = torch.tensor([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])



# ============================================
# STEP 2: A REUSABLE TRAINING FUNCTION
# ============================================
# We will train TWO different models the exact same way,
# so we put the training loop in one function and reuse it.

def train(model, epochs=5000, lr=0.1):
    loss_fn = nn.MSELoss()                              # (predicted - actual)^2
    optimizer = optim.Adam(model.parameters(), lr=lr)   # Adam = reliable optimizer

    for epoch in range(epochs):
        y_pred = model(X)              # 1) forward pass  -> predictions
        loss = loss_fn(y_pred, y)      # 2) how wrong are we?
        optimizer.zero_grad()          # 3) clear old gradients
        loss.backward()                # 4) compute new gradients
        optimizer.step()               # 5) update the weights

    return loss.item()                 # final loss after training


# small helper to print predictions nicely (rounded)
def show(model):
    with torch.no_grad():
        preds = model(X)
    for i in range(4):
        inp = X[i].tolist()
        want = int(y[i].item())
        got = preds[i].item()
        print(f"   input {inp} -> want {want} , model says {got:.2f}")


# ============================================
# STEP 3: MODEL A -> THE LINEAR MODEL (WILL FAIL)
# ============================================
# 2 inputs -> 1 output. Just ONE straight line. No hidden layer.

# torch.manual_seed(1)   # fixed seed so results are repeatable
# linear_model = nn.Linear(2, 1)

# print("=" * 50)
# print("MODEL A: LINEAR (no hidden layer)")
# print("=" * 50)
# linear_loss = train(linear_model)
# print(f"Final loss = {linear_loss:.4f}   <- stuck! (0.25 = pure guessing)")
# show(linear_model)
# print("Result: FAILS. It outputs ~0.50 for everything because")
# print("        no single straight line can separate XOR.\n")


# ============================================
# STEP 4: MODEL B -> THE MULTILAYER MODEL (WILL WORK)
# ============================================
# Same 2 inputs -> but now we add a HIDDEN LAYER of 8 neurons,
# and a NON-LINEAR activation (ReLU) between the layers.
#
# nn.Sequential just stacks layers in order:
#   Linear(2 -> 8)  ->  ReLU  ->  Linear(8 -> 1)
#
# The ReLU is the secret sauce. Without it, stacking linear
# layers would just collapse back into one line and fail again.

torch.manual_seed(1)   # same seed -> fair comparison
multilayer_model = nn.Sequential(
    nn.Linear(2, 8),   # hidden layer: 2 inputs -> 8 hidden neurons
    nn.ReLU(),         # non-linearity: lets the model bend, not just draw a line
    nn.Linear(8, 1),   # output layer: 8 hidden -> 1 output
)

print("=" * 50)
print("MODEL B: MULTILAYER (1 hidden layer + ReLU)")
print("=" * 50)
mlp_loss = train(multilayer_model)
print(f"Final loss = {mlp_loss:.4f}   <- solved!")
show(multilayer_model)
print("Result: WORKS. Predictions snap to 0 and 1 correctly.\n")
