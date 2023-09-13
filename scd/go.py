import requests
import bs4

ROOT = "https://www.mathe2.uni-bayreuth.de"


def main():
    r = requests.get(ROOT + "/markus/reggraphs.html")
    soup = bs4.BeautifulSoup(r.content, "html.parser")
    table = soup.find_all("table")[0]
    links = table.find_all("a")

    for link in links:
        url = link["href"]
        r = requests.get(ROOT + url)
        sublinks = bs4.BeautifulSoup(r.content, "html.parser").find_all("a")
        for sublink in sublinks:
            suburl = sublink["href"]
            if suburl.endswith(".scd"):
                r = requests.get(
                    ROOT + "/markus/REGGRAPHS/" + suburl, allow_redirects=True
                )
                with open(suburl, "wb") as f:
                    f.write(r.content)


if __name__ == "__main__":
    main()
