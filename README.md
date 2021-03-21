# National Assembly session speech generator

An of-the-shelf OpenAI GPT 2 model trained on the speeches from the National Assembly of Serbia, scraped from the official website.

The GPT2 model is based on the code provided at: https://towardsdatascience.com/train-gpt-2-in-your-own-language-fc6ad4d60171

## Running the model

### Scraping
Scrape the speeches data by running:

``` console
python3 scraper.py
```

As of today, there are about 2000 sessions and scraping takes around 1 hour.

### Generating dataset

Convert the scraped data into training-friendly data:

``` console
python3 raw_sessions_data_to_text.py
```

This operation is not time intensive.

### Training and running the model

Finally run the generator file, which will try to load trained model, or train it if saved model is not found. During training all intermediates, as well as the final model, will be saved to avoid later recalculation.

Pass the initial word list to the text generator as command line parameters and the generator will load/train the model and print out the continuation of the speech:

``` console
python3 gpt_model.py Some initial words here
```

As currently parameterized, using all provided data (partially filtered inside the Generating dataset step) and 3 epochs, the training takes around 18 hours on the one GTX1070 GPU.

The saved model loads quickly and the generator takes a few seconds to generate a 100 word output.
