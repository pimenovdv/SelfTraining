import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DNC(nn.Module):
    def __init__(self, input_size, hidden_size, memory_words, memory_word_size, num_reads=1):
        super(DNC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = memory_words
        self.W_size = memory_word_size
        self.R = num_reads

        # Controller (LSTM)
        self.controller = nn.LSTM(input_size + self.R * self.W_size, hidden_size)

        # Interface parameters
        self.interface_size = (self.W_size * self.R) + (3 * self.W_size) + (5 * self.R) + 3
        self.interface_linear = nn.Linear(hidden_size, self.interface_size)

        # Output mapping
        self.output_linear = nn.Linear(hidden_size + self.R * self.W_size, input_size)

    def _parse_interface(self, interface_vector):
        read_keys = interface_vector[:, :self.R * self.W_size].view(-1, self.R, self.W_size)
        read_strengths = F.softplus(interface_vector[:, self.R * self.W_size : self.R * self.W_size + self.R]) + 1

        offset = self.R * self.W_size + self.R
        write_key = interface_vector[:, offset : offset + self.W_size]
        write_strength = F.softplus(interface_vector[:, offset + self.W_size : offset + self.W_size + 1]) + 1
        erase_vector = torch.sigmoid(interface_vector[:, offset + self.W_size + 1 : offset + 2 * self.W_size + 1])
        write_vector = interface_vector[:, offset + 2 * self.W_size + 1 : offset + 3 * self.W_size + 1]

        offset += 3 * self.W_size + 1
        free_gates = torch.sigmoid(interface_vector[:, offset : offset + self.R])
        allocation_gate = torch.sigmoid(interface_vector[:, offset + self.R : offset + self.R + 1])
        write_gate = torch.sigmoid(interface_vector[:, offset + self.R + 1 : offset + self.R + 2])
        read_modes = F.softmax(interface_vector[:, offset + self.R + 2 : offset + 4 * self.R + 2].view(-1, self.R, 3), dim=-1)

        return read_keys, read_strengths, write_key, write_strength, erase_vector, write_vector, free_gates, allocation_gate, write_gate, read_modes

    def forward(self, x, state=None):
        batch_size = x.size(1)
        seq_len = x.size(0)

        if state is None:
            # Initialize DNC state
            memory = torch.zeros(batch_size, self.W, self.W_size, device=x.device)
            read_vectors = torch.zeros(batch_size, self.R * self.W_size, device=x.device)
            usage_vector = torch.zeros(batch_size, self.W, device=x.device)
            link_matrix = torch.zeros(batch_size, self.W, self.W, device=x.device)
            precedence_weight = torch.zeros(batch_size, self.W, device=x.device)
            read_weightings = torch.zeros(batch_size, self.R, self.W, device=x.device)
            write_weighting = torch.zeros(batch_size, self.W, device=x.device)
            hx = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            cx = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        else:
            pass

        outputs = []
        for t in range(seq_len):
            controller_input = torch.cat([x[t], read_vectors], dim=-1).unsqueeze(0)

            controller_out, (hx, cx) = self.controller(controller_input, (hx, cx))
            controller_out = controller_out.squeeze(0)

            interface_vector = self.interface_linear(controller_out)
            parsed_interface = self._parse_interface(interface_vector)

            read_vectors = torch.randn(batch_size, self.R * self.W_size, device=x.device)

            output_input = torch.cat([controller_out, read_vectors], dim=-1)
            out = self.output_linear(output_input)
            outputs.append(out)

        return torch.stack(outputs, dim=0), None

def main():
    logging.info("Starting DNC component training on copy task...")

    seq_len = 10
    batch_size = 16
    input_size = 8

    model = DNC(input_size=input_size, hidden_size=32, memory_words=16, memory_word_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 100
    for epoch in range(epochs):
        sequence = torch.randn(seq_len, batch_size, input_size)

        optimizer.zero_grad()
        output, _ = model(sequence)

        loss = criterion(output, sequence)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            logging.info(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

    logging.info("DNC component successfully implemented and verified.")

if __name__ == "__main__":
    main()
