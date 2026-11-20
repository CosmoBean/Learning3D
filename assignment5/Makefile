REPORT_MD := index.md
REPORT_HTML := index.html
MD2HTML := ../scripts/md2html.sh

.PHONY: all html clean

all: html

html: $(REPORT_HTML)

$(REPORT_HTML): $(REPORT_MD)
	bash $(MD2HTML) $(REPORT_MD) $(REPORT_HTML)

clean:
	rm -f $(REPORT_HTML)
