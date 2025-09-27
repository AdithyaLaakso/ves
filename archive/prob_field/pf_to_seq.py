import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Tuple

from greek_char_prob_field import greek_char_prob_field as ProbField
from passage_to_prob_field import passage_to_prob_field_list
from etc import read_random_file
from etc import read_file_num

class GreekCharProbDataset(Dataset):
    def __init__(self, sequences_with_targets, max_seq_length=100):
        self.max_seq_length = max_seq_length
        self.samples = []
        
        for sequence, target_passage in sequences_with_targets:
            if len(sequence) > 0 and len(target_passage) > 0:
                # Convert sequence of probability fields to input tensor
                input_probs = []
                for prob_field in sequence:
                    input_probs.append(prob_field.probs)
                
                # Pad or truncate input sequence
                if len(input_probs) > max_seq_length:
                    input_probs = input_probs[:max_seq_length]
                else:
                    # Pad with zeros
                    while len(input_probs) < max_seq_length:
                        input_probs.append(np.zeros(48))  # 48 = vocab_size
                
                # Convert target passage to character indices
                target_indices = []
                for char in target_passage:
                    target_idx = ProbField.get_char_idx(char)
                    if target_idx is not None and target_idx > 0:  # Only include valid characters
                        target_indices.append(target_idx)
                
                # Pad or truncate target sequence to match input length
                if len(target_indices) > max_seq_length:
                    target_indices = target_indices[:max_seq_length]
                else:
                    # Pad with a special padding token (use 0 or create a special token)
                    while len(target_indices) < max_seq_length:
                        target_indices.append(0)  # Padding token
                
                if len(target_indices) > 0:
                    self.samples.append((
                        np.array(input_probs),
                        np.array(target_indices),
                        len(sequence),  # actual input length
                        len([c for c in target_passage if (ProbField.get_char_idx(c) is not None and ProbField.get_char_idx(c) > 0)])
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target_seq, input_len, target_len = self.samples[idx]
        return (
            torch.FloatTensor(input_seq),
            torch.LongTensor(target_seq),
            torch.LongTensor([input_len]),
            torch.LongTensor([target_len])
        )

class GreekTextCollapseModel(nn.Module):
    def __init__(self, vocab_size=48, hidden_size=256, num_layers=3, dropout=0.3):
        super(GreekTextCollapseModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder LSTM - processes the probability field sequence
        self.encoder_lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Decoder LSTM - generates the output sequence
        self.decoder_lstm = nn.LSTM(
            input_size=vocab_size + hidden_size * 2,  # +2*hidden for bidirectional encoder
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 3, 1)  # decoder + encoder (bidirectional)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size * 3, vocab_size)  # decoder + context
        self.dropout = nn.Dropout(dropout)
        
        # Embedding for decoder input (convert indices to embeddings)
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, encoder_input, target_sequence=None, max_length=None):
        batch_size = encoder_input.size(0)
        device = encoder_input.device
        
        # Encode the probability field sequence
        encoder_output, (encoder_h, encoder_c) = self.encoder_lstm(encoder_input)
        # encoder_output: (batch_size, seq_len, hidden_size * 2) due to bidirectional
        
        # Initialize decoder hidden state with encoder's final state
        # For bidirectional encoder, we need to combine forward and backward states
        decoder_h = encoder_h[-self.num_layers:].contiguous()  # Take last num_layers
        decoder_c = encoder_c[-self.num_layers:].contiguous()
        
        if target_sequence is not None:
            # Training mode - use teacher forcing
            target_len = target_sequence.size(1)
            outputs = []
            
            # Start with a start token (use index 0 or create a special start token)
            decoder_input = torch.zeros(batch_size, 1, self.vocab_size).to(device)
            
            for t in range(target_len):
                # Calculate attention
                decoder_h_expanded = decoder_h[-1].unsqueeze(1).expand(-1, encoder_output.size(1), -1)
                attention_input = torch.cat([decoder_h_expanded, encoder_output], dim=2)
                attention_weights = F.softmax(self.attention(attention_input), dim=1)
                context = torch.sum(attention_weights * encoder_output, dim=1, keepdim=True)
                
                # Decoder input: previous character embedding + context
                lstm_input = torch.cat([decoder_input, context], dim=2)
                
                # LSTM forward
                decoder_output, (decoder_h, decoder_c) = self.decoder_lstm(lstm_input, (decoder_h, decoder_c))
                
                # Output projection
                output = torch.cat([decoder_output, context], dim=2)
                output = self.dropout(output)
                output = self.output_projection(output)  # (batch_size, 1, vocab_size)
                outputs.append(output)
                
                # Teacher forcing: use actual target as next input
                if t < target_len - 1:
                    next_char_idx = target_sequence[:, t].unsqueeze(1)
                    decoder_input = self.embedding(next_char_idx).float()
            
            return torch.cat(outputs, dim=1)  # (batch_size, target_len, vocab_size)
        
        else:
            # Inference mode - generate sequence
            if max_length is None:
                max_length = encoder_input.size(1)
            
            outputs = []
            decoder_input = torch.zeros(batch_size, 1, self.vocab_size).to(device)
            
            for t in range(max_length):
                # Calculate attention
                decoder_h_expanded = decoder_h[-1].unsqueeze(1).expand(-1, encoder_output.size(1), -1)
                attention_input = torch.cat([decoder_h_expanded, encoder_output], dim=2)
                attention_weights = F.softmax(self.attention(attention_input), dim=1)
                context = torch.sum(attention_weights * encoder_output, dim=1, keepdim=True)
                
                # Decoder input: previous character embedding + context
                lstm_input = torch.cat([decoder_input, context], dim=2)
                
                # LSTM forward
                decoder_output, (decoder_h, decoder_c) = self.decoder_lstm(lstm_input, (decoder_h, decoder_c))
                
                # Output projection
                output = torch.cat([decoder_output, context], dim=2)
                output = self.dropout(output)
                output = self.output_projection(output)  # (batch_size, 1, vocab_size)
                outputs.append(output)
                
                # Use predicted character as next input
                predicted_idx = torch.argmax(output, dim=2)
                decoder_input = self.embedding(predicted_idx).float()
            
            return torch.cat(outputs, dim=1)  # (batch_size, max_length, vocab_size)

def train_model(model, train_loader, num_epochs=100, learning_rate=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (encoder_input, target_seq, input_lengths, target_lengths) in enumerate(train_loader):
            encoder_input = encoder_input.to(device)
            target_seq = target_seq.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(encoder_input, target_sequence=target_seq)
            
            # Calculate loss
            # output: (batch_size, target_len, vocab_size)
            # target_seq: (batch_size, target_len)
            
            output_flat = output.view(-1, output.size(-1))  # (batch_size * target_len, vocab_size)
            target_flat = target_seq.view(-1)  # (batch_size * target_len)
            
            loss = criterion(output_flat, target_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (excluding padding tokens)
            predictions = torch.argmax(output_flat, dim=1)
            mask = target_flat != 0  # Non-padding tokens
            if mask.sum() > 0:
                correct_predictions += ((predictions == target_flat) & mask).sum().item()
                total_predictions += mask.sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

def predict_passage(model, prob_field_sequence, device='cpu', max_length=None):
    """
    Predict the collapsed text from a sequence of probability fields
    """
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Convert probability fields to tensor
        input_probs = []
        for prob_field in prob_field_sequence:
            input_probs.append(prob_field.probs)
        
        # Pad sequence if necessary
        max_seq_length = 100  # Should match training
        if len(input_probs) > max_seq_length:
            input_probs = input_probs[:max_seq_length]
        else:
            while len(input_probs) < max_seq_length:
                input_probs.append(np.zeros(48))
        
        input_tensor = torch.FloatTensor(input_probs).unsqueeze(0).to(device)
        
        # Generate prediction
        if max_length is None:
            max_length = len(prob_field_sequence)
        
        output = model(input_tensor, max_length=max_length)
        predictions = torch.argmax(output, dim=2).squeeze(0).cpu().numpy()
        
        # Convert predictions to characters
        predicted_chars = []
        for idx in predictions:
            if idx >= 0:  # Skip padding tokens
                char = ProbField.get_char_from_idx(predictions[idx])
                predicted_chars.append(char)
        predicted_passage = ''.join(predicted_chars)
    
    return predicted_passage

# Update your main function
def main():
    print("Generating training data...")
    
    # Generate training sequences
    training_data = []
    num_training_sequences = 1500  # Increase this for better training
    
    for i in range(num_training_sequences):
        print(f"Generating training sequence #{i}")
        sequence, target_passage = generate_training_sequence(i)
        training_data.append((sequence, target_passage))
    
    # Create dataset and dataloader
    dataset = GreekCharProbDataset(training_data, max_seq_length=150)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)  # Smaller batch size for sequence-to-sequence
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Create model
    model = GreekTextCollapseModel(vocab_size=48, hidden_size=256, num_layers=3)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    train_model(model, train_loader, num_epochs=10, learning_rate=0.001, device=device)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE - TESTING MODEL")
    print("="*50)
    
    # Test the model
    for i in range(5):
        test_sequence, expected_passage = generate_training_sequence(i + num_training_sequences)
        predicted_passage = predict_passage(model, test_sequence, device=device)
        
        print(f"\nTest {i+1}:")
        print(f"Expected:  '{expected_passage[:50]}{'...' if len(expected_passage) > 50 else ''}'")
        print(f"Predicted: '{predicted_passage[:50]}{'...' if len(predicted_passage) > 50 else ''}'")
        
        # Calculate character-level accuracy
        min_len = min(len(expected_passage), len(predicted_passage))
        if min_len > 0:
            correct_chars = sum(1 for j in range(min_len) if expected_passage[j] == predicted_passage[j])
            accuracy = correct_chars / min_len
            print(f"Character accuracy: {accuracy:.2%}")
    
    return model

def generate_training_sequence(num: int) -> Tuple[List, str]:
    passage = read_file_num("./Greek_Raw_Cleaned/", num)
    print(f"\t Passage: {passage[:5]}")
    seq = passage_to_prob_field_list(passage)
    return (seq, passage)

if __name__ == "__main__":
    model = main()
