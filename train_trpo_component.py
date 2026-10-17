import numpy as np
import os
import argparse

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class TRPONetwork:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2. / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W_actor = np.random.randn(hidden_dim, action_dim) * np.sqrt(2. / hidden_dim)
        self.b_actor = np.zeros(action_dim)
        self.W_critic = np.random.randn(hidden_dim, 1) * np.sqrt(2. / hidden_dim)
        self.b_critic = np.zeros(1)

    def get_action_and_value(self, state):
        self.x = state
        self.z1 = np.dot(self.x, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z_actor = np.dot(self.a1, self.W_actor) + self.b_actor
        self.probs = softmax(self.z_actor)
        self.z_critic = np.dot(self.a1, self.W_critic) + self.b_critic
        self.value = self.z_critic[0, 0]
        return self.probs, self.value, self.a1

    def backward_critic(self, state, td_error):
        _, _, a1 = self.get_action_and_value(state)
        d_z_critic = np.array([[-td_error]])
        d_W_critic = np.dot(a1.T, d_z_critic)
        d_b_critic = np.sum(d_z_critic, axis=0)
        d_a1 = np.dot(d_z_critic, self.W_critic.T)
        d_z1 = d_a1 * relu_deriv(self.z1)
        d_W1 = np.dot(self.x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)
        return d_W1, d_b1, d_W_critic, d_b_critic

    def get_flat_params(self):
        return np.concatenate([self.W_actor.flatten(), self.b_actor.flatten()])

    def set_flat_params(self, params):
        idx = 0
        w_shape = self.W_actor.shape
        w_size = np.prod(w_shape)
        self.W_actor = params[idx:idx+w_size].reshape(w_shape)
        idx += w_size
        b_shape = self.b_actor.shape
        b_size = np.prod(b_shape)
        self.b_actor = params[idx:idx+b_size].reshape(b_shape)

    def policy_loss_and_grad(self, states, actions, old_probs_all, advantages):
        loss = 0
        flat_grad = np.zeros_like(self.get_flat_params())
        N = len(states)
        for i in range(N):
            probs, _, a1 = self.get_action_and_value(states[i])
            curr_prob = probs[0, actions[i]]
            ratio = curr_prob / (old_probs_all[i] + 1e-8)
            loss += (ratio * advantages[i]) / N

            grad_ratio = advantages[i] / (old_probs_all[i] + 1e-8)
            d_z_actor = -probs.copy()
            d_z_actor[0, actions[i]] += 1.0
            d_z_actor *= (grad_ratio * curr_prob)

            d_W_actor = np.dot(a1.T, d_z_actor)
            d_b_actor = np.sum(d_z_actor, axis=0)

            grad = np.concatenate([d_W_actor.flatten(), d_b_actor.flatten()])
            flat_grad += grad / N

        return loss, flat_grad

    def kl_divergence(self, old_probs_all, current_probs_all):
        kl = 0
        for old, new in zip(old_probs_all, current_probs_all):
            kl += np.sum(old * np.log((old + 1e-8) / (new + 1e-8)))
        return kl / len(old_probs_all)

    def fisher_vector_product(self, states, p):
        N = len(states)
        fvp = np.zeros_like(p)
        for i in range(N):
            probs, _, a1 = self.get_action_and_value(states[i])

            for a in range(probs.shape[1]):
                prob_a = probs[0, a]
                d_z_actor = -probs.copy()
                d_z_actor[0, a] += 1.0

                d_W_actor = np.dot(a1.T, d_z_actor)
                d_b_actor = np.sum(d_z_actor, axis=0)

                grad = np.concatenate([d_W_actor.flatten(), d_b_actor.flatten()])

                grad_dot_p = np.dot(grad, p)
                fvp += prob_a * grad * grad_dot_p / N

        return fvp + 0.1 * p

class SimpleEnv:
    def __init__(self):
        self.state = 2
    def step(self, action):
        if action == 0:
            self.state -= 1
        else:
            self.state += 1
        if self.state == 4:
            return self.state, 1.0, True
        elif self.state == 0:
            return self.state, -1.0, True
        else:
            return self.state, -0.1, False
    def reset(self):
        self.state = 2
        return self.state

def one_hot(state, size=5):
    vec = np.zeros((1, size))
    vec[0, state] = 1.0
    return vec

def conjugate_gradient(Avp, b, nsteps, residual_tol=1e-10):
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rdotr = r.dot(r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / (p.dot(_Avp) + 1e-8)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = r.dot(r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def linesearch(model, states, actions, old_probs_val, old_probs_all, advantages, max_kl, expected_improve_rate, step_dir, max_backtracks=10, accept_ratio=0.1):
    initial_params = model.get_flat_params()
    initial_loss, _ = model.policy_loss_and_grad(states, actions, old_probs_val, advantages)

    for stepfrac in 0.5**np.arange(max_backtracks):
        new_params = initial_params + stepfrac * step_dir
        model.set_flat_params(new_params)

        new_loss, _ = model.policy_loss_and_grad(states, actions, old_probs_val, advantages)

        new_probs_all = []
        for s in states:
            p, _, _ = model.get_action_and_value(s)
            new_probs_all.append(p)

        kl = model.kl_divergence(old_probs_all, new_probs_all)

        actual_improve = new_loss - initial_loss
        expected_improve = expected_improve_rate * stepfrac

        if kl <= max_kl and actual_improve > 0:
            return True

    model.set_flat_params(initial_params)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr_critic", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_kl", type=float, default=0.01)
    args = parser.parse_args()

    np.random.seed(42)
    model = TRPONetwork(state_dim=5, hidden_dim=args.hidden_dim, action_dim=2)
    env = SimpleEnv()

    print(f"Training TRPO with hidden_dim={args.hidden_dim}, epochs={args.epochs}, gamma={args.gamma}, max_kl={args.max_kl}")

    for ep in range(args.epochs):
        state = env.reset()
        done = False
        states, actions, rewards, old_probs_val, old_probs_all, values = [], [], [], [], [], []

        while not done and len(states) < 20:
            x = one_hot(state)
            probs, value, _ = model.get_action_and_value(x)
            action = np.random.choice(2, p=probs[0])
            next_state, reward, done = env.step(action)

            states.append(x)
            actions.append(action)
            rewards.append(reward)
            old_probs_val.append(probs[0, action])
            old_probs_all.append(probs)
            values.append(value)

            state = next_state

        if not done:
            _, next_value, _ = model.get_action_and_value(one_hot(state))
        else:
            next_value = 0.0

        returns, advantages = [], []
        for i in range(len(states)):
            G, pw = 0, 0
            for r in rewards[i:]:
                G += (args.gamma**pw) * r
                pw += 1
            if not done:
                G += (args.gamma**pw) * next_value
            returns.append(G)
            advantages.append(G - values[i])

        adv = np.array(advantages)
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

        dW1_c_total, db1_c_total, dWc_total, dbc_total = 0, 0, 0, 0
        for i in range(len(states)):
            td_error = values[i] - returns[i]
            dW1, db1, dWc, dbc = model.backward_critic(states[i], td_error)
            dW1_c_total += dW1
            db1_c_total += db1
            dWc_total += dWc
            dbc_total += dbc

        model.W1 += args.lr_critic * dW1_c_total / len(states)
        model.b1 += args.lr_critic * db1_c_total / len(states)
        model.W_critic += args.lr_critic * dWc_total / len(states)
        model.b_critic += args.lr_critic * dbc_total / len(states)

        _, policy_grad = model.policy_loss_and_grad(states, actions, old_probs_val, adv)

        def FVP(p):
            return model.fisher_vector_product(states, p)

        step_dir = conjugate_gradient(FVP, policy_grad, nsteps=10)

        shs = 0.5 * np.dot(step_dir, FVP(step_dir))
        if shs > 0:
            lm = np.sqrt(shs / args.max_kl)
            fullstep = step_dir / lm
            expected_improve = np.dot(policy_grad, fullstep)

            success = linesearch(model, states, actions, old_probs_val, old_probs_all, adv, args.max_kl, expected_improve, fullstep)

        if ep % (args.epochs // 10) == 0 or ep == args.epochs - 1:
            print(f"Episode {ep}: Total Reward = {sum(rewards):.2f}")

    state = env.reset()
    done, success = False, False
    print("\nEvaluating final policy:")
    while not done:
        probs, value, _ = model.get_action_and_value(one_hot(state))
        action = np.argmax(probs[0])
        print(f"State: {state} -> Action: {action} (Probs: {probs[0]}, Value: {value:.2f})")
        state, _, done = env.step(action)
        if state == 4:
            success = True

if __name__ == "__main__":
    main()
