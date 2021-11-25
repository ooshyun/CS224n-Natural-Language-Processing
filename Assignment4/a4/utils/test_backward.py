import torch

def example_attribute():
    # Creating the graph
    x = torch.tensor(1.0, requires_grad = True)
    y = torch.tensor(2.0)
    z = x * y

    # Displaying
    for i, name in zip([x, y, z], "xyz"):
        print(f"{name}\ndata: {i.data}\nrequires_grad: {i.requires_grad}\n\
    grad: {i.grad}\ngrad_fn: {i.grad_fn}\nis_leaf: {i.is_leaf}\n")

def example_req_grad():
    # Creating the graph
    x = torch.tensor(1.0, requires_grad = True)
    # Check if tracking is enabled
    print(x.requires_grad) #True
    y = x * 2
    print(y.requires_grad) #True

    with torch.no_grad():
        # Check if tracking is enabled
        y = x * 2
        print(y.requires_grad) #False

def example_backward():
    # Creating the graph
    x = torch.tensor(1.0, requires_grad = True)
    z = x ** 3
    z.backward() #Computes the gradient 
    print(x.grad.data) #Prints '3' which is dz/dx 



if __name__=='__main__':
    example_backward()