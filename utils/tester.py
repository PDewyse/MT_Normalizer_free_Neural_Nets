import torch
import os
import tqdm
from utils.metrics import calculate_metrics

class Tester:
    def __init__(self, model, criterion, dataloader, save_dir, log=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.log = log
        if self.log:
            header = f"{self.model.__class__.__name__}\t{self.save_dir}\n"
            columns = "\t".join(["test_loss", "accuracy", "balanced accuracy", "precision", "recall", "f1", "roc_auc"]) + "\n"
            with open(os.path.join(self.save_dir, "test_log.txt"), 'w') as f:
                f.write(header + columns)
            print(f"Testing log created at {os.path.join(self.save_dir, 'test_log.txt')}")

    def test(self):
        self.model.eval()
        test_loss = 0.0
        all_probabilities = []
        all_labels = []
        with torch.no_grad():
            with tqdm.tqdm(total=len(self.dataloader), desc="Testing") as pbar:
                for inputs, labels in self.dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item()

                    # Metrics calculation
                    probabilities = torch.softmax(outputs, dim=1)
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    pbar.update(1)
        
        # pass the output probabilities to be able to calculate roc_auc
        test_loss /= len(self.dataloader)
        metrics = calculate_metrics(all_probabilities, all_labels)
        self._log_metrics(test_loss, metrics)
        return test_loss, metrics
    
    def _log_metrics(self, loss, metrics):
        log_data = {"test_loss": loss, **metrics}
        log_message = "\t".join([f"{k}\t{v}" for k, v in log_data.items()]) + "\n"
        if self.log:
            with open(os.path.join(self.save_dir, "test_log.txt"), 'a') as f:
                f.write(log_message)
        else:
            print("\n" + log_message)