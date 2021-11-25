# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

conda env create --file local_env.yml

# Colab 
- mount the google drive
- train the model
- change
    - download package
        - sacrebleu
        - sentencepiece
    - vocab.get_vocab_list
        - source path
    - run.sh
        - path_curr

# Colab Tip
- Check the GPU in colab  
    !nvidia-smi

- Prevent to stop colab
    1. Open the Colab in Chrome
    2. F12 or Ctrl+Shift+i in Linus, Windows / Command + Option + I in Mac
    3. Enter the code in the Console
    - Code
        ```
        function ClickConnect() { 
                        var buttons = document.querySelectorAll("colab-dialog.yes-no-dialog paper-button#cancel"); 
                        buttons.forEach(function(btn) { 
                                        btn.click(); 
                        }); 
                        console.log("Automatically reconnect every 1 minute"); 
                        document.querySelector("#top-toolbar > colab-connect-button").click(); 
        } 
        setInterval(ClickConnect,1000*60);
        ```