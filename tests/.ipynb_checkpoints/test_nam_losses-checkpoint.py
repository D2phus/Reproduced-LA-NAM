import torch
import pytest 


testdata = [
    (torch.randn(10, 3)),
    (torch.randn(2, 3)), 
]

@pytest.mark.parametrize('f', testdata)
def test_concurvity_loss(f):  
    def method1(f): 
        batch_size, in_features = f.shape
        if batch_size < 2 or in_features < 2: 
            # no enough samples / features for correlation computing 
            return torch.zeros(1)
        
        corr_matrix = torch.corrcoef(torch.transpose(f, 1, 0)).abs()
        R = torch.triu(corr_matrix, diagonal=1).sum()
        R /= (in_features*(in_features-1)/2)
        return R
    
    def method2(f): 
        batch_size, in_features = f.shape
        if batch_size < 2 or in_features < 2: 
            # no enough samples / features for correlation computing 
            return torch.zeros(1)
        mean = f.mean(dim=0)
        f -= mean 

        R = list()
        for i in range(in_features):
            for j in range(i+1, in_features):
                a, b = f[:, i], f[:, j]
                r = (a@b).sum()
                r /= a.square().sum().sqrt()* b.square().sum().sqrt()
                r = r.abs()
                R.append(r)

        R = sum(R) / (in_features*(in_features-1)/2)
        return R
    
    def method3(f, eps=1e-12):
        batch_size, in_features = f.shape
        if batch_size < 2 or in_features < 2: 
            # no enough samples / features for correlation computing 
            return torch.zeros(1)
        
        std = f.std(dim=0)
        std = std*std.reshape(-1, 1)
        cov = torch.cov(f.T)
        
        corr_matrix = cov/(std + eps) 
        corr_matrix = torch.where(std==0.0, 0.0, corr_matrix)
        
        R = torch.triu(corr_matrix.abs(), diagonal=1).sum()
        R /= (in_features*(in_features-1)/2)
        return R
        
    assert torch.equal(method1(f), method2(f))
    assert torch.equal(method3(f), method2(f))