from torch.utils.data import Dataset
# dataset类
class GraphDataset(Dataset):
    def __init__(self, pyg_datas):
        super().__init__()
        # print("pyg_datas:",pyg_datas)
        self.pyg_datas = pyg_datas
    #获取子图feature和子图label
    def __getitem__(self, index):
        #获取子图
        return self.pyg_datas[index]

    def __len__(self):
        return len(self.pyg_datas)