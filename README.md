# ScrapeGPT 
## Telegram Bot for Web Content Analysis and Question Answering

ScrapeGPT is a Telegram bot designed to scrape and analyze websites, then answer questions based on the scraped content. The bot utilizes advanced natural language processing techniques to provide accurate responses to user queries.

## Features

- **Web Scraping**: Automatically scrapes text from provided URLs, including PDF files.
- **Context Retrieval**: Utilizes embeddings and retrieval models to extract relevant context from scraped content.
- **Question Answering**: Generates answers to user questions based on the retrieved context.
- **Robots.txt Parsing**: Respects website's robots.txt to avoid scraping restricted areas.
- **Database Management**: Stores scraped content in a database for future reference and quick access.
- **Proxy Support**: Uses rotating proxies to bypass geo-restrictions and anonymize requests.
- **LLM-based**: Supports both public and local LLMs.

## Getting Started

To get started with the bot, follow these steps:

1. Clone the repository to your local machine:
```bash
git clone https://github.com/LexiestLeszek/scrapeGPT.git
cd scrapeGPT
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your Telegram bot by obtaining a token from the [BotFather](https://core.telegram.org/bots#botfather), replace `API_TOKEN` in the script with your actual token and don't forget bot's telegram nickname.
4. Run the bot server script:
```bash
python scrapeGPT.py
```

## Usage

Once the bot is running, interact with it on Telegram:

1. Find the bot by its name and open the chat with the bot
2. Send `/start` command to initialize the bot.
3. Provide a website URL to begin analysis.
4. Ask a question regarding the analyzed content, and the bot will respond with an answer.

## Contributing

Contributions are welcome! To contribute:

- Fork the repository.
- Make changes in your fork.
- Submit a pull request with a clear description of your changes.

## License

This project is licensed under the terms of the MIT license. See the LICENSE file for details.

## Contact

If you have any questions or suggestions, feel free to open an issue on GitHub.

## No liability for the Developer

- Usage of this software for attacking targets without prior mutual consent is illegal.
- It is the end user's responsibility to obey all applicable local, state and federal laws.
- Developers of this software assume no liability and are not responsible for any misuse or damage caused by this program
by third parties using the software in violation of laws and regulations.
