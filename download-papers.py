import urllib.request
import xml.etree.ElementTree as ET
import os
import time


def download_arxiv_papers(query, num_papers=50, save_dir="data/"):
    """
    Automatically search arXiv and download PDFs.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1: search arXiv API for papers
    print(f"Searching arXiv for: {query}")

    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": num_papers,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    url = base_url + urllib.parse.urlencode(params) # converts the params dictionary into a URL string
    response = urllib.request.urlopen(url)
    content = response.read().decode("utf-8")

    # 2: parse the XML response to get paper IDs
    # arXiv returns results in XML format
    root = ET.fromstring(content)     # reads the XML
    namespace = "{http://www.w3.org/2005/Atom}"  # in arxiv format every tag is prefixed with this namespace

    entries = root.findall(f"{namespace}entry")    # extracts each paper 'entry'
    print(f"Found {len(entries)} papers")

    # 3: download each PDF
    downloaded = 0
    for entry in entries:

        # get paper ID from the URL
        paper_id = entry.find(f"{namespace}id").text
        # paper_id looks like: http://arxiv.org/abs/2301.12345v1
        # we need just: 2301.12345
        arxiv_id = paper_id.split("/abs/")[-1].split("v")[0]

        # get paper title for display
        title = entry.find(f"{namespace}title").text.strip()
        title = title.replace("\n", " ")

        # build PDF download URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        save_path = os.path.join(save_dir, f"{arxiv_id}.pdf")

        # skip if already downloaded
        if os.path.exists(save_path):
            print(f"Already exists: {arxiv_id}")
            downloaded += 1
            continue

        try:
            print(f"Downloading [{downloaded+1}/{num_papers}]: {title[:60]}...")
            urllib.request.urlretrieve(pdf_url, save_path)
            downloaded += 1

            # wait 3 seconds between downloads
            # arXiv asks not to hammer their servers.If you download too fast, they may block your IP.
            time.sleep(3)

        except Exception as e:
            print(f"Failed to download {arxiv_id}: {e}")
            continue

    print(f"\nDone! Downloaded {downloaded} papers to {save_dir}")

if __name__ == "__main__":
    download_arxiv_papers(
        query="extractive summarization scientific papers",
        num_papers=50,
        save_dir="data/"
    )