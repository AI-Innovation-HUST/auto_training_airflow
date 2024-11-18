import torch
import pandas as pd
import os
import torch.nn as nn
from model import *
from dataloader_v2 import *
from sklearn.model_selection import train_test_split

def load_data(df, test_size=0.1, val_size=0.2):
   try:
       # Bỏ cột ts vì không cần cho việc training
       if 'ts' in df.columns:
           df = df.drop('ts', axis=1)
           
       os.makedirs("results", exist_ok=True)
       
       # Split data
       train_val_df, test_df = train_test_split(df, test_size=test_size, shuffle=False, random_state=42)
       train_df, val_df = train_test_split(train_val_df, test_size=val_size, shuffle=False, random_state=42)

       train_dataset = CoinDataset(train_df)
       val_dataset = CoinDataset(val_df.reset_index(drop=True))
       test_dataset = CoinDataset(test_df.reset_index(drop=True))
       
       print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
       return train_dataset, val_dataset, test_dataset
   except Exception as e:
       print(f"Lỗi trong load_data: {e}")
       raise e

def evaluate(model, epoch, criterion_high, criterion_low, data_source, dev):
   model.eval()
   total_loss = 0
   total_acc_h = 0
   total_acc_l = 0
   count = 0
   
   with torch.no_grad():
       for data, targets in data_source:
           targets = targets.view(-1, 2).to(dev)
           output = model(data.to(dev))
           
           # Xử lý output
           if isinstance(output, tuple):
               high, low = output
           else:
               # Nếu model trả về tensor đơn
               high = output[:, 0].view(-1, 1)
               low = output[:, 1].view(-1, 1)
           
           # Tính loss
           loss_high = criterion_high(high.float(), targets[:, 0].float())
           loss_low = criterion_low(low.float(), targets[:, 1].float())
           total_loss += (loss_high.item() + loss_low.item()) / 2
           
           # Tính accuracy
           score_h, score_l = compute_acc(high, low, targets)
           total_acc_h += score_h.item() if isinstance(score_h, torch.Tensor) else score_h
           total_acc_l += score_l.item() if isinstance(score_l, torch.Tensor) else score_l
           count += 1
   
   avg_loss = total_loss / count if count > 0 else float('inf')
   avg_acc_h = total_acc_h / count if count > 0 else 0
   avg_acc_l = total_acc_l / count if count > 0 else 0
   
   print(f"Epoch {epoch} - Validation: Loss = {avg_loss:.4f}, High Acc = {avg_acc_h:.4f}, Low Acc = {avg_acc_l:.4f}")
   
   return avg_loss, avg_acc_h, avg_acc_l

def training_transformer():
   try:
       print("Bắt đầu training...")
       df = pd.read_csv("raw_data/scaled_data.csv")
       if df.empty:
           print("Không có dữ liệu để train")
           return None

       dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
       print(f"Using device: {dev}")

       # Load và split data
       train_dataset, val_dataset, test_dataset = load_data(df)
       
       # Print một dòng dữ liệu để kiểm tra
       print("\nMột dòng dữ liệu mẫu:")
       sample_row = df.iloc[0]
       print(sample_row)
       
       train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
       val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)
       test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
       
       print("Train dataset size:", len(train_dataset))
       print("Validation dataset size:", len(val_dataset))
       print("Test dataset size:", len(test_dataset))
       
       # Check input size from data
       vocab_size = df.shape[1] - 1  # trừ đi cột ts
       print(f"Vocabulary size from data: {vocab_size}")

       # Initialize model với vocab_size từ data
       model = Transformer(n_blocks=4, d_model=16, n_heads=8, d_ff=256, dropout=0.5, vocab_size=vocab_size)
       model.to(dev)

       criterion_high = nn.MSELoss()
       criterion_low = nn.MSELoss()
       lr = 0.0001
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
       epochs = 100
       best_val_loss = float('inf')

       for epoch in range(epochs):
           model.train()
           count = 0
           cum_loss_h = 0
           cum_loss_l = 0
           accs_h = 0
           accs_l = 0

           for batch in train_loader:
               data, targets = batch
               targets = targets.view(-1,2).to(dev)
               output = model(data.to(dev))
               
               # Xử lý output của model
               if isinstance(output, tuple):
                   high, low = output
               else:
                   high = output[:, 0].view(-1, 1)
                   low = output[:, 1].view(-1, 1)
               
               # Đảm bảo kích thước đúng
               high = high.view(-1, 1)
               low = low.view(-1, 1)
               targets = targets.view(-1, 2)
               
               loss_high = criterion_high(high.float(), targets[:,0].float().view(-1, 1))
               loss_low = criterion_low(low.float(), targets[:,1].float().view(-1, 1))
               total_loss = 0.5 * loss_high + 0.5 * loss_low
               
               score_h, score_l = compute_acc(high, low, targets)
               accs_h += score_h.item() if isinstance(score_h, torch.Tensor) else score_h
               accs_l += score_l.item() if isinstance(score_l, torch.Tensor) else score_l
               cum_loss_h += loss_high.item()
               cum_loss_l += loss_low.item()
               
               total_loss.backward()
               optimizer.step()
               model.zero_grad()
               optimizer.zero_grad()
               count += 1

           print(f"Epoch {epoch}:")
           print(f"Loss high: {cum_loss_h/count:.4f}")
           print(f"Loss low: {cum_loss_l/count:.4f}")
           print(f"ACC high train: {accs_h/count:.4f}")
           print(f"ACC low train: {accs_l/count:.4f}")

           print("------START EVAL------")
           eval_loss, acc_h, acc_l = evaluate(model, epoch, criterion_high, criterion_low, val_loader, dev=dev)
           
           if eval_loss < best_val_loss:
               best_val_loss = eval_loss
               torch.save({
                   'epoch': epoch,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'loss': best_val_loss,
               }, "results/evalModel_best.pth")
               print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

           scheduler.step()
       
       return best_val_loss

   except Exception as e:
       print(f"Lỗi trong training_transformer: {e}")
       raise e

if __name__ == "__main__":
   print("Bắt đầu test quá trình training...")
   try:
       val_loss = training_transformer()
       if val_loss is not None:
           print(f"\nTraining thành công! Best validation loss: {val_loss:.4f}")
       else:
           print("\nTraining thất bại!")
   except Exception as e:
       print(f"Lỗi trong quá trình test: {e}")