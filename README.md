# AIPersonCount
AI Person Count with Open CV , Tensforflow with minimum number of code lines.

Using this few lines , you can do person detection and person count on any youtube video, 

To change the youtube video link, go to python file and replace the youtube link and your youtube file will run and you can see the person count. 
The accuracy is not upto mark. I will add more repos with similar usecases. 

This program detect common objects as Person , Car, Trucks etc and also shows the count of people in each Frame. 
This will need Tensorflow on MacOS, as of Apr 2023, this needs Python < 3.11 , I did with Python 3.10.9


![Screenshot 2023-04-06 at 4 14 54 PM](https://user-images.githubusercontent.com/26647401/230355115-6c6c6bf2-85c7-4a0e-84b9-ecfc983a6ee2.png)

## PersonaAgent Demo

This repository also includes `persona_agent_demo.py`, a demonstration of the
PersonaAgent framework for personalized question answering. The script loads a
quantized LLM from Hugging Face and requires a valid Hugging Face token for
gated models such as *Mistral-7B-Instruct*.

To run the demo in Google&nbsp;Colab:

1. Install the required packages:

   ```bash
   pip install transformers torch accelerate bitsandbytes sentence-transformers faiss-cpu
   ```

2. Obtain a Hugging Face access token and set it in the environment before
   running the script:

   ```bash
   import os
   os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
   ```

   You can generate a token from <https://huggingface.co/settings/tokens> after
   accepting the model's license.

3. Execute the script:

   ```bash
   python persona_agent_demo.py
   ```
