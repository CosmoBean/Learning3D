"""Screenshot an HTML viewer for each option of a <select id=...> (default: scene)."""
import argparse

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("html")
parser.add_argument("prefix")
parser.add_argument("--select", default="scene")
parser.add_argument("--count", type=int, default=5)
parser.add_argument("--geometry", action="store_true", help="Also cycle the GT/prediction selector")
args = parser.parse_args()

with sync_playwright() as p:
    page = p.chromium.launch().new_page(viewport={"width": 1400, "height": 1000})
    page.goto(f"file://{args.html}")
    page.wait_for_timeout(3000)
    values = page.eval_on_selector_all(
        f"#{args.select} option", "o => o.map(x => x.value)"
    )
    for i, value in enumerate(values[: args.count]):
        page.select_option(f"#{args.select}", value)
        page.wait_for_timeout(2000)
        if not args.geometry:
            page.screenshot(path=f"{args.prefix}_{i}.png", full_page=True)
            print("saved", f"{args.prefix}_{i}.png")
            continue
        options = page.eval_on_selector_all(
            "#geometry option", "o => o.map(x => [x.value, x.textContent])"
        )
        for value, label in options:
            page.select_option("#geometry", value)
            page.wait_for_timeout(2000)
            name = f"{args.prefix}_{i}_{label.split()[0].lower()}.png"
            page.screenshot(path=name, full_page=True)
            print("saved", name, label)
