import gradio as gr
import os
import time
import requests, json, os, re, ollama, time, logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from fp.fp import FreeProxy
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings

# Proxy init
def get_proxy():
    print("Starting proxy ...")
    proxy_url = FreeProxy(country_id=['US','CA','FR','NZ','SE','PT','CZ','NL','ES','SK','UK','PL','IT','DE','AT','JP'],https=True,rand=True,timeout=3).get()
    proxy_obj = {
        "server": proxy_url,
        "username": "",
        "password": ""
    }

    print(f"Proxy generated: {proxy_url}")
    
    return proxy_obj

def scrape_webpages(urls,proxy):
    print("Scraping text from webpages from each of the links ...")
    scraped_texts = []
    for url in urls:
        try:
            if url.endswith('.pdf'):
                response = requests.get(url, proxies=proxy)
                reader = PdfReader(BytesIO(response.content))
                number_of_pages = len(reader.pages)

                for p in range(number_of_pages):

                    page = reader.pages[p]
                    text = page.extract_text()
                    scraped_texts.append(text)
            else:
                page = requests.get(url,proxies=proxy)
                soup = BeautifulSoup(page.content, 'html.parser')
                text = ' '.join([p.get_text() for p in soup.find_all('p')])
                scraped_texts.append(text)

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    
    all_scraped_text = '\n'.join(scraped_texts)
    print("Finished scraping the text from webpages!")
    return all_scraped_text

def get_domain(url):
    return urlparse(url).netloc

def get_robots_file(url,proxy):
    robots_url = urljoin(url, '/robots.txt')
    try:
        response = requests.get(robots_url,proxies=proxy)
        return response.text
    except Exception as e:
        print(f"Error fetching robots.txt: {e}")
        return None

def parse_robots(content):
    # This function assumes simple rules without wildcards, comments, etc.
    # For a full parser, consider using a library like robotparser.
    disallowed = []
    for line in content.splitlines():
        if line.startswith('Disallow:'):
            path = line[len('Disallow:'):].strip()
            disallowed.append(path)
    return disallowed

def is_allowed(url, disallowed_paths, base_domain):
    parsed_url = urlparse(url)
    if parsed_url.netloc != base_domain:
        return False
    for path in disallowed_paths:
        if parsed_url.path.startswith(path):
            return False
    return True

def scrape_site_links(start_url, proxy):
    visited_links = set()
    not_visited_links = set()
    to_visit = [start_url]
    base_domain = get_domain(start_url)
    disallowed_paths = parse_robots(get_robots_file(start_url, proxy))
    last_found_time = time.time()  # Track the last time a link was found

    while to_visit:
        # Break the loop if  30 seconds have passed without finding a new link
        if time.time() - last_found_time >  15:
            print("FINISHED scraping the links")
            break

        current_url = to_visit.pop(0)
        if current_url not in visited_links and is_allowed(current_url, disallowed_paths, base_domain):
            visited_links.add(current_url)
            try:   
                print(f"{current_url}")
                response = requests.get(current_url, proxies=proxy)
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    new_url = urljoin(current_url, link['href'])
                    if new_url not in visited_links:
                        to_visit.append(new_url)
                        last_found_time = time.time()  # Update the last found time
            except Exception as e:
                print(f" !!! COULD NOT VISIT: {current_url}")
                not_visited_links.add(current_url)

    return visited_links

def analyze_website(start_url):
    global shared_result
    
    proxy = get_proxy()
    
    # Scrape all the links from the given start URL using the proxy
    all_links = scrape_site_links(start_url, proxy)

    # Scrape the content from all the links obtained, using the proxy
    full_text = scrape_webpages(all_links, proxy)
    
    shared_result = full_text
    
    txt = "Website Analyzed!"
    
    return txt

def ask_questions(question_text):
    global shared_result
    if shared_result is None:
        return "No result available yet."
    else:
    
        #full_text = " ".join(line for line in full_text.splitlines() if line)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        paper_chunks = text_splitter.create_documents([shared_result])
        
        qdrant = Qdrant.from_documents(
            documents=paper_chunks,
            embedding=HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2'),
            path="./tmp/local_qdrant",
            collection_name="data",
        )
        retriever = qdrant.as_retriever()
        
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        ollama_llm = "qwen:0.5b"
        model = ChatOllama(model=ollama_llm)
        
        chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | model
            | StrOutputParser()
        )
        
        class Question(BaseModel):
            __root__: str
        
        chain = chain.with_types(input_type=Question)
        result = chain.invoke(question_text)
        return result

# Create the interfaces for each task
iface1 = gr.Interface(fn=analyze_website, inputs="text", outputs="text",title="Enter website URL")
iface2 = gr.Interface(fn=ask_questions, inputs="text", outputs="text",title="Ask questions to the website")

# Combine the interfaces into a TabbedInterface
tabbed_interface = gr.TabbedInterface([iface1, iface2], ["URL Input", "QnA with Website"])

# Launch the combined interface
tabbed_interface.launch()
