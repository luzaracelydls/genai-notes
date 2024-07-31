## Lesson 5: Text Generation with Vertex AI

#### Project environment setup

- Load credentials and relevant Python Libraries


```python
from utils import authenticate
credentials, PROJECT_ID = authenticate()
```


```python
REGION = 'us-central1'
```

### Prompt the model
- We'll import a language model that has been trained to handle a variety of natural language tasks, `text-bison@001`.
- For multi-turn dialogue with a language model, you can use, `chat-bison@001`.


```python
import vertexai
vertexai.init(project=PROJECT_ID, 
              location=REGION, 
              credentials = credentials)
```


```python
from vertexai.language_models import TextGenerationModel
```


```python
generation_model = TextGenerationModel.from_pretrained(
    "text-bison@001")
```

#### Question Answering
- You can ask an open-ended question to the language model.


```python
prompt = "I'm a high school student. \
Recommend me a programming activity to improve my skills."
```


```python
print(generation_model.predict(prompt=prompt).text)
```

    * **Write a program to solve a problem you're interested in.** This could be anything from a game to a tool to help you with your studies. The important thing is that you're interested in the problem and that you're motivated to solve it.
    * **Take a programming course.** There are many online and offline courses available, so you can find one that fits your schedule and learning style.
    * **Join a programming community.** There are many online and offline communities where you can connect with other programmers and learn from each other.
    * **Read programming books and articles.** There is a


#### Classify and elaborate
- For more predictability of the language model's response, you can also ask the language model to choose among a list of answers and then elaborate on its answer.


```python
prompt = """I'm a high school student. \
Which of these activities do you suggest and why:
a) learn Python
b) learn Javascript
c) learn Fortran
"""
```


```python
print(generation_model.predict(prompt=prompt).text)
```

    I would suggest learning Python. Python is a general-purpose programming language that is easy to learn and has a wide range of applications. It is used in a variety of fields, including web development, data science, and machine learning. Python is also a popular language for beginners, as it has a large community of support and resources available.


#### Extract information and format it as a table


```python
prompt = """ A bright and promising wildlife biologist \
named Jesse Plank (Amara Patel) is determined to make her \
mark on the world. 
Jesse moves to Texas for what she believes is her dream job, 
only to discover a dark secret that will make \
her question everything. 
In the new lab she quickly befriends the outgoing \
lab tech named Maya Jones (Chloe Nguyen), 
and the lab director Sam Porter (Fredrik Johansson). 
Together the trio work long hours on their research \
in a hope to change the world for good. 
Along the way they meet the comical \
Brenna Ode (Eleanor Garcia) who is a marketing lead \
at the research institute, 
and marine biologist Siri Teller (Freya Johansson).

Extract the characters, their jobs \
and the actors who played them from the above message as a table
"""
```


```python
response = generation_model.predict(prompt=prompt)

print(response.text)
```

    | Character | Job | Actor |
    |---|---|---|
    | Jesse Plank | Wildlife Biologist | Amara Patel |
    | Maya Jones | Lab Tech | Chloe Nguyen |
    | Sam Porter | Lab Director | Fredrik Johansson |
    | Brenna Ode | Marketing Lead | Eleanor Garcia |
    | Siri Teller | Marine Biologist | Freya Johansson |


- You can copy-paste the text into a markdown cell to see if it displays a table.

| Character | Job | Actor |
|---|---|---|
| Jesse Plank | Wildlife Biologist | Amara Patel |
| Maya Jones | Lab Tech | Chloe Nguyen |
| Sam Porter | Lab Director | Fredrik Johansson |
| Brenna Ode | Marketing Lead | Eleanor Garcia |
| Siri Teller | Marine Biologist | Freya Johansson |

### Adjusting Creativity/Randomness
- You can control the behavior of the language model's decoding strategy by adjusting the temperature, top-k, and top-n parameters.
- For tasks for which you want the model to consistently output the same result for the same input, (such as classification or information extraction), set temperature to zero.
- For tasks where you desire more creativity, such as brainstorming, summarization, choose a higher temperature (up to 1).


```python
temperature = 0.0
```


```python
prompt = "Complete the sentence: \
As I prepared the picture frame, \
I reached into my toolkit to fetch my:"
```


```python
response = generation_model.predict(
    prompt=prompt,
    temperature=temperature,
)
```


```python
print(f"[temperature = {temperature}]")
print(response.text)
```

    [temperature = 0.0]
    As I prepared the picture frame, I reached into my toolkit to fetch my hammer.



```python
temperature = 1.0
```


```python
response = generation_model.predict(
    prompt=prompt,
    temperature=temperature,
)
```


```python
print(f"[temperature = {temperature}]")
print(response.text)
```

    [temperature = 1.0]
    As I prepared the picture frame, I reached into my toolkit to fetch my screwdriver. A screwdriver is a tool that is used to drive screws into or remove them from a surface. There are many different types of screwdrivers, each with its own specific purpose. The most common type of screwdriver is the slotted screwdriver, which has a thin, flat blade that fits into the slot of a screw. Other types of screwdrivers include the Phillips head screwdriver, which has a cross-shaped blade, and the Torx head screwdriver, which has a six-pointed star-shaped blade.


#### Top P
- Top p: sample the minimum set of tokens whose probabilities add up to probability `p` or greater.
- The default value for `top_p` is `0.95`.
- If you want to adjust `top_p` and `top_k` and see different results, remember to set `temperature` to be greater than zero, otherwise the model will always choose the token with the highest probability.


```python
top_p = 0.2
```


```python
prompt = "Write an advertisement for jackets \
that involves blue elephants and avocados."
```


```python
response = generation_model.predict(
    prompt=prompt, 
    temperature=0.9, 
    top_p=top_p,
)
```


```python
print(f"[top_p = {top_p}]")
print(response.text)
```

    [top_p = 0.2]
    **Introducing the new Blue Elephant Avocado Jacket!**
    
    This jacket is the perfect way to show your love of both blue elephants and avocados. It's made of soft, durable fabric and features a cozy lining. The front has a large blue elephant print, and the back has a smaller avocado print.
    
    This jacket is perfect for any occasion. Wear it to a concert, a party, or just a casual day out. You're sure to turn heads wherever you go!
    
    **Order your Blue Elephant Avocado Jacket today!**


#### Top k
- The default value for `top_k` is `40`.
- You can set `top_k` to values between `1` and `40`.
- The decoding strategy applies `top_k`, then `top_p`, then `temperature` (in that order).


```python
top_k = 20
top_p = 0.7
```


```python
response = generation_model.predict(
    prompt=prompt, 
    temperature=0.9, 
    top_k=top_k,
    top_p=top_p,
)
```


```python
print(f"[top_p = {top_p}]")
print(response.text)
```

    [top_p = 0.7]
    **Introducing the new Blue Elephant Avocado Jacket!**
    
    This jacket is the perfect way to show your love for both blue elephants and avocados. It's made of soft, durable fabric and features a fun, embroidered design.
    
    The Blue Elephant Avocado Jacket is perfect for any occasion. Wear it to a casual day out, or dress it up for a night on the town. You'll be sure to turn heads wherever you go.
    
    **Order your Blue Elephant Avocado Jacket today!**
    
    * * *
    
    **Here are some of the benefits of owning a Blue Elephant Avocado Jacket:**
    
    *



```python

```


```python

```
