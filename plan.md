# Dataset creation class

Create an InMemory PyTorch Geometric (PyG) dataset class. Call it as you want. Something like `MolDataset` would do. The train and test (we will skip val) sets must be saved in 2 different subfolders of `data`. It should have the follwoing structure:

- data:
    - train:
        - raw:
            - `data.pkl(csv)` - this is a pickle or csv file with the train SMILES and potency values
    - test:
        - raw:
            - `data.pkl(csv)` - this is a pickle or csv file with the test SMILES and potency values

Check [this](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html) how to create such a dataset. We did a more complicated version of that on day 2 of the classes. You can see how it's done there as well. You should initialise a python `logger` in the file where you define the dataset, so all info will be logged and shown in the terminal. Check how to do it. It's 2 lines of code. You can use that logger for debugging purposes too. Implement the methods needed for the dataset to work.

- raw_file_names() - basically it's those `data.pkl(or csv)` file names in the folders specified above.
- processed_file_names() - you should save the processed data into 1 file, you can call it just `data.pt` or something like that.
- download() - skip that and just write something like that:
    ```python
    def download(self):
        # Use the initialised above logger to output that to terminal
        logger.warning("Skipping download because it's not implemented. You should download your data manually")

    ```
- process():
    It must read the data from the file and output a graph. So you need to unpickle or just read a csv into some object like a list to get the SMILES. Then using that list, get the following output for each SMILES in the list:

    - data.u_chem - a torch tensor of ChemBERTa embeddings of shape (1, 256). Call ChemBERTa on SMILES to get it, use `pipe` method written in the `starting_point.ipynb` notebook for simplicity.
    - data.u_dm - a torch tensor of concatenated datamol features of shape (1, `F_dm`), where `F_dm` is the size of the concatenated feature vector. Use datamol on SMILES to get it.
    - data.x - a torch tensor of node features extracted from Graphium of shape (`Num_nodes`, `F_x`). Use datamol/graphium on SMILES to get it.
    - data.pos - a torch tensor of 3D coordinates of shape (`Num_nodes`, `Num_conformers`, 3). So the idea is that for 1 SMILES representation we will get N number of conformers and save their related coordinates in that `data.pos` tensor object. The shape is set like that for the convenience of batching operations. It means, that for each node we will get `Num_conformers` x,y,z coordinates. Use datamol on SMILES to get it. The function is in the notebook `smiles_to_3d.ipynb`. You can check the notebook and take that function directly, but probably you'd need to reshape the output to make it of a proper shape.  Let's set `num_conformers` via the `config`, which is located in `config/__init__.py`. You can add there a new variable called `NUM_CONFORMERS` or so. All vars in that file must be CAPITALIZED like that. You should then import that config in your file like that:
        ```python
        import config as cfg
        ```
        This will import it and you can access your variables there like that:
        ```
        my_var = cfg.MY_VAR
        ```
        Check `models/mpnn/gat.py` to see how I do it.
    - data.y - a torch tensor of our targets.

**NB! Make sure the tensors are of the proper data.type like `torch.float`. All the shapes above are given for 1 graph, so 1 SMILES representation. We will process a list of them and save them as a list of PyG `Data` objects, which represent our graphs with the attributes listed above.**

- pre_transform() = `Compose([Scale(), ComleteGraph(), ConcatenateGlobal()])`, where `Compose()` is a PyG transform to compose several transforms together. 
    - Import and apply PyG transform class `Scale()` to `data.u_dm` and `data.pos`, since all other features are scaled already. Check how it's done in the tutorial. You need to get new scaled attributes `data.u_dm` and `data.pos`.
    - Check GDL day 3 colab and find a transform class that creates a complete graph (all nodes connected to all nodes). Copy paste that transform class and add to the list of transforms to compose together.
    - Create your own transform called `ConcatenateGlobal` that would concatenate 2 global attributes `data.u_chem` and `data.u_dm` into a new one called `data.u`, then delete them from the `data` object. So the `data` will look like that:
        - data.x: dim = [Num_nodes, F_x]
        - data.u: dim = [1, 256 + F_dm]
        - data.pos: dim = [Num_nodes, Num_conformers, 3]
        - data.y: dim = [1,]
        We need to have this transform to concatenate already scaled features to the embeddings of the ChemBERTa, that's why this transform should be called after `Scale()`
    
    Feel free to create a separate file called `transforms.py` in `data` directory and define them there and then import from there into `dataset.py` that you're creating.

- transform - None
    Ignore that method.


Save the python file that creates the dataset in `data` directory, you can call it `dataset.py` or something like that and make that directory a python module, creating there an empty python file called  `__init__.py`


Run the file on train and test sets and make sure it works as expected! You can do it in a notebook for example and save that notebook for testing data related stuff in `notebooks` directory. Try creating a PyG dataloader object with that dataset and sample a batch of graphs from it. It must work. Check how we created dataloaders in the GDL class.

# FeatureEmbedder class.

This class will be the first block that will be called in the whole model. It will embed all our features to some dimension. You should implement the following methods:
- `__init__()`
    - This method should take the following args:
        - input_mol_emb_dim = `cfg.IN_MOL_DIM` (check what should be used there after concatenation of molecular features)
        - input_node_emb_dim = `cfg.NODE_DIM` (check the dimension of the node features that you get from Graphium)
        - out_emb_dim = `cfg.OUT_EMB_DIM` (maybe 32/64 that you can set in the `config/__init__.py`)

    Maybe, we will need more args, but so far that's it

    `__init__()` should initialise 2 MLPs:
    - `mlp_mol_emb` - MLP to embed the concatenated global molecular features `data.u`. Use PyG `MLP` class with `torch.silu()` non-linearity and PyG `LayerNorm()` normalization. You can specify them as the arguments of the `MLP` class. You can set the number of layers to 1-2 hidden layers with not too many neurons. 
    - `mlp_x_emb` - MLP to embed `data.x` features. Use PyG `MLP` class with `torch.silu()` non-linearity and PyG `LayerNorm()` normalization. You can specify them as the arguments of the `MLP` class. You can set the number of layers to 1-2 hidden layers with not too many neurons.

    Both MLPs should produce the output of the same dimension (`cfg.OUT_EMB_DIM`)


- forward(batch_data) 

    - This method works with the batched data that we get from a dataloader that uses the dataset class you'll create in the step 1. A batch of data will be a PyG `Batch` object that we worked during our GDL classes. It will have all the attributes we set above in the dataset + data.batch attribute. It will perform the following actions:
        - Embed the global concatenated molecular `data.u` features into some dimension `cfg.OUT_EMB_DIM` with the `mlp_mol_emb` and save the output back into `data.u`.
        - Embed `data.x` into some dimension `cfg.OUT_EMB_DIM` with `mlp_x_emb` and save the output to `data.x`
        - Return our modified `batch_data` object

    So this method applies both MLPs to our input features and embeds them into the same dimension. You can test this class also in the data notebook that you'll create for testing the datasets and dataloaders.
