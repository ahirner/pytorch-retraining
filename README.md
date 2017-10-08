# pytorch-retraining
Transfer Learning shootout for PyTorch's model zoo (torchvision).

* **Load** any pretrained model with custom final layer (num_classes) from PyTorch's model zoo in one line
```python
model_pretrained, diff = load_model_merged('inception_v3', num_classes)
```

* **Retrain** minimal (as inferred on load) or a custom amount of layers on multiple GPUs
```python
final_param_names = [d[0] for d in diff]
stats = train_eval(model_pretrained, trainloader, testloader, final_params_names)
```

* **Chart** `training_time`, `evaluation_time` (fps), top-1 `accuracy` for varying levels of retraining depth (shallow, deep and from scratch)

|  ![chart](https://raw.githubusercontent.com/ahirner/pytorch-retraining/master/results/diagram_bees.png) | 
|:---:|
| *Transfer learning on example dataset [Bee vs Ants](http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)* with 2xK80 GPUs|
