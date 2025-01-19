import wandb
import tqdm
import os
import torch
from utils.metrics import calculate_metrics
from utils.signal_logger import SignalLogger

class Trainer:
    def __init__(self, model, criterion, optimizer, dataloader, save_dir, scheduler=None, save_every=1, log=False, log_signal=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.save_every = save_every
        self.log = log
        self.log_signal = log_signal
        self.update_freq = 0  # 0.1 for Updating signal statistics every 10% of the batches
        if self.log_signal:
            self.signal_logger = SignalLogger(self.model)
        # TODO: fix that stupid tqdm from making multiple progress bars
        # TODO: make a logger class (not the one for signal stats)

    def train(self):
        self.model.train()
        epoch_stats = None
        if self.log_signal:
            hooks = self.signal_logger.register_hooks()  # Register hooks for forward and backward pass
            # epoch_stats = {}  # Dictionary to store signal statistics for each batch

        running_loss = 0.0
        total_batches = len(self.dataloader)
        update_interval = max(1, int(total_batches * self.update_freq))  # Ensure at least one update per epoch
        
        with tqdm.tqdm(total=total_batches, desc="Batch progress") as pbar:
            for i, (inputs, labels) in enumerate(self.dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if self.log_signal and (i + 1) % update_interval == 0:
                    self.signal_logger.extract_weights()
                    self.signal_logger.update_running_stats()
                    # batch_stats = self.signal_logger.get_signal_statistics()
                    # epoch_stats[i] = batch_stats

                pbar.set_postfix({"batch_loss": loss.item()})
                pbar.update(1)

        if self.scheduler:
            self.scheduler.step()

        signal_stats = None
        if self.log_signal:
            for hook in hooks:
                hook.remove()  # Unregister hooks
            # Get signal statistics
            signal_stats = self.signal_logger.get_signal_statistics()
            self.signal_logger.clear()

        return running_loss / total_batches, signal_stats, epoch_stats

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_probabilities = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                # Metrics calculation
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # pass the output probabilities to be able to calculate roc_auc
        val_loss /= len(val_loader)
        metrics = calculate_metrics(all_probabilities, all_labels)
        return val_loss, metrics
    
    def train_model(self, num_epochs, val_loader=None):
        best_accuracy = 0.0
        
        with tqdm.tqdm(total=len(self.dataloader)*num_epochs, desc="Epoch progress") as pbar:
            for epoch in range(num_epochs):
                train_loss, signal_stats, epoch_stats = self.train()
                if val_loader:
                    val_loss, metrics = self.validate(val_loader)
                    pbar.set_postfix({"train_loss": train_loss, "val_loss": val_loss, "accuracy" : metrics["accuracy"]})
    
                    self._log_metrics_and_save_model(epoch, train_loss, val_loss, metrics, signal_stats, epoch_stats)
                    if metrics["accuracy"] > best_accuracy:
                        best_accuracy = metrics["accuracy"]
                        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))

                    pbar.update(len(self.dataloader))
    
                else:
                    pbar.set_postfix({"train_loss": train_loss})
                    pbar.update(len(self.dataloader))
    
                    self._log_metrics_and_save_model(epoch, train_loss, None, None, signal_stats)

        pbar.close()

    def _log_metrics_and_save_model(self, epoch, train_loss, val_loss, metrics, signal_stats, epoch_stats=None):
        log_data = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        
        if signal_stats:
            log_data.update(signal_stats)
            
        if epoch_stats:
            epoch_stats_path = os.path.join(self.save_dir, "epoch_signal_stats.txt")
            if not os.path.exists(epoch_stats_path):
                with open(epoch_stats_path, 'w') as f:
                    f.write("epoch\tbatch\tlayer_name\tmean_act\tstd_act\tmean_weight\tstd_weight\tmean_grad\tstd_grad\n")
            with open(epoch_stats_path, 'a') as f:
                for batch, stats in epoch_stats.items():
                    for layer, stat in stats.items():
                        f.write(f"{epoch}\t{batch}\t{layer}\t{stat['mean_act']}\t{stat['std_act']}\t{stat['mean_weight']}\t{stat['std_weight']}\t{stat['mean_grad']}\t{stat['std_grad']}\n")

        log_file = os.path.join(self.save_dir, "training_log.txt")
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                header = f"{self.model.__class__.__name__}\t{self.save_dir}\n"
                column_names = "\t".join([str(k) for k in log_data.keys()]) + "\n"
                f.write(header + column_names)

        # Append to the log file
        with open(log_file, 'a') as f:
            f.write("\t".join([str(v) for v in log_data.values()]) + "\n")

        log_data = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        if self.log:
            wandb.log({k: v for k, v in log_data.items()}) # note, without signal stats
        else:
            print("\n" + "\t".join([f"{k}\t{v}" for k, v in log_data.items()]) + "\n")

        if (epoch + 1) % self.save_every == 0:
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_epoch{epoch + 1}_acc{metrics['accuracy']}.pt"))
