"""Animate one example research question through the feature-discovery pipeline."""

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "docs" / "assets"
WIDTH, HEIGHT = 1080, 480
INK, MUTED, LINE = "#17212b", "#66727e", "#dce1e5"
BLUE, PALE, GREEN = "#2458a6", "#edf3fb", "#28704c"
DURATIONS = [4.0, 5.6, 4.8, 4.8, 4.0, 4.8, 5.6, 5.6]
MOTION_SCALE = 1.5
TRANSITION_SECONDS = 0.8
FRAME_MS = 80
QUESTION = ("What emotions are described", "in psychology research?")
FEATURES = ["Happiness", "Feeling happy", "Sadness", "Sleep quality"]
PAPERS = ("Zhang et al.", "Garcia et al.", "Patel et al.", "Wilson et al.")
COLLECTION_GRID = ((790, 156), (925, 156), (790, 286), (925, 286))
COLLECTION_ORDER = (1, 0, 3, 2)


@lru_cache(maxsize=32)
def font(size):
    for name in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError("Install DejaVu Sans to render the documentation animation.")


def ease(value):
    return 1 - (1 - max(0, min(1, value))) ** 3


class Slide:
    def __init__(self, image=None):
        self.image = image if image is not None else Image.new("RGB", (WIDTH, HEIGHT), "white")
        self.draw = ImageDraw.Draw(self.image)

    def text(self, x, y, text, size=22, color=INK):
        self.draw.text((round(x), round(y)), text, font=font(size), fill=color)

    def line(self, points, color=LINE, width=1):
        self.draw.line(points, fill=color, width=width)

    def rect(self, box, fill=None, outline=None):
        self.draw.rectangle(tuple(round(v) for v in box), fill=fill, outline=outline)

    def check(self, x, y, selected):
        self.rect(
            (x, y, x + 21, y + 21), fill=BLUE if selected else "white", outline=BLUE if selected else LINE
        )
        if selected:
            self.line([(x + 4, y + 11), (x + 9, y + 16), (x + 17, y + 5)], "white", 2)

    def paper(self, x, y, scale=1, color=BLUE):
        w, h, fold = 32 * scale, 42 * scale, 8 * scale
        self.draw.polygon(
            [(x, y), (x + w - fold, y), (x + w, y + fold), (x + w, y + h), (x, y + h)],
            fill="white",
            outline=color,
        )
        self.line([(x + w - fold, y), (x + w - fold, y + fold), (x + w, y + fold)], color)
        for offset in [17, 24, 31]:
            self.line([(x + 7 * scale, y + offset * scale), (x + w - 7 * scale, y + offset * scale)], color)

    def arrow(self, start, end, progress=1):
        x, y = start
        ex, ey = end
        p = ease(progress)
        tip = (x + (ex - x) * p, y + (ey - y) * p)
        self.line([start, tip], BLUE, 2)
        if p > 0.8:
            self.line([(tip[0] - 7, tip[1] - 5), tip, (tip[0] - 7, tip[1] + 5)], BLUE, 2)

    def entrance(self, painter, progress, dx=0, dy=16):
        if progress <= 0:
            return
        layer = Slide()
        painter(layer)
        p = ease(progress)
        shifted = Image.new("RGB", (WIDTH, HEIGHT), "white")
        shifted.paste(layer.image, (round(dx * (1 - p)), round(dy * (1 - p))))
        mask = ImageChops.invert(shifted.convert("L")).point(lambda value: round(min(255, value * 15) * p))
        self.image.paste(shifted, (0, 0), mask)
        self.draw = ImageDraw.Draw(self.image)


def question(s, t):
    s.text(64, 40, "1. Research question", 30)


def question_panel(image, progress=1):
    """Move the opening question into a fixed panel, independent of scene fades."""
    p = ease(progress)
    x, y = 140 + 490 * p, 185 - 167 * p
    width, height = 800 - 414 * p, 150 - 48 * p
    inset, text_size = 24 - 8 * p, round(34 - 13 * p)
    s = Slide(image)
    s.rect((x, y, x + width, y + height), fill=PALE)
    s.line([(x, y), (x, y + height)], BLUE, 3)
    s.text(x + inset, y + 12, "FEATURE QUESTION", round(14 - 2 * p), BLUE)
    for index, line in enumerate(QUESTION):
        s.text(x + inset, y + 39 - 6 * p + index * (44 - 16 * p), line, text_size)
    return s.image


def collection_paper(index, t):
    """Fill the right cell before the left so arrivals never cross a parked PDF."""
    slot = COLLECTION_ORDER[index]
    target_x, target_y = COLLECTION_GRID[slot]
    progress = (t - 0.4 - index * 0.55) / 0.5
    if progress <= 0:
        return None
    p = ease(progress)
    x, y = 650 + (target_x - 650) * p, 150 + index * 59 + (target_y - 150 - index * 59) * p
    return x, y, 0.8 + 0.8 * p, slot, progress >= 1


def collect(s, t):
    s.text(64, 40, "2. Collect papers", 30)
    s.text(64, 216, "Emotions in", 25)
    s.text(64, 253, "psychology", 25)
    s.arrow((266, 247), (377, 247), t / 0.5)
    for i, name in enumerate(["OpenAlex", "Semantic Scholar", "Crossref", "Europe PMC"]):
        y = 150 + i * 59
        s.entrance(
            lambda layer, y=y, name=name: layer.text(406, y, name, 23),
            (t - 0.2 - i * 0.12) / 0.35,
            dx=-18,
            dy=0,
        )
        position = collection_paper(i, t)
        if position:
            x, y, scale, slot, arrived = position
            s.paper(x, y, scale)
            if arrived:
                label_x = x + 25.6 - font(16).getlength(PAPERS[slot]) / 2
                s.text(label_x, y + 78, PAPERS[slot], 16, MUTED)


def select(s, t):
    s.text(64, 40, "3. Select PDFs", 30)
    for i, name in enumerate(PAPERS):
        y = 166 + i * 66
        s.check(100, y + 3, i < 2 and t > 0.3 + i * 0.35)
        s.text(142, y, name, 23, INK if i < 2 else MUTED)
        s.line([(100, y + 48), (550, y + 48)])
        if i < 2 and t > 0.85 + i * 0.4:
            x = 727 + i * 153
            p = min(1, (t - 0.85 - i * 0.4) / 0.9)
            s.entrance(lambda layer, x=x: layer.paper(x, 192, 2), (t - 0.85 - i * 0.4) / 0.35)
            s.text(x + 32 - font(16).getlength(PAPERS[i]) / 2, 286, PAPERS[i], 16, MUTED)
            s.rect((x - 9, 315, x + 73, 318), fill=LINE)
            s.rect((x - 9, 315, x - 9 + 82 * p, 318), fill=GREEN if p == 1 else BLUE)
    s.text(770, 361, "Selected PDFs", 21, MUTED)


def extract(s, t):
    s.text(64, 40, "4. Extract features", 30)
    s.paper(153, 156, 2.3)
    s.text(100, 278, PAPERS[0], 18, MUTED)
    s.text(100, 312, "“Participants described feeling", 22)
    s.text(100, 346, "happy or sad.”", 22)
    s.arrow((385, 245), (495, 245), (t - 0.2) / 0.6)
    for i, name in enumerate(FEATURES):
        y = 152 + i * 65
        s.entrance(
            lambda layer, y=y, name=name: (
                layer.text(551, y, name, 25),
                layer.line([(551, y + 46), (980, y + 46)]),
            ),
            (t - 0.55 - i * 0.35) / 0.4,
            dx=25,
            dy=0,
        )


def filter_features(s, t):
    s.text(64, 40, "5. Filter features", 30)
    for i, name in enumerate(FEATURES):
        y = 152 + i * 65
        if i < 3:
            s.text(200, y, name, 27)
            s.line([(200, y + 48), (875, y + 48)])
            if t > 0.45 + i * 0.2:
                s.check(830, y + 5, True)
        else:
            p = ease((t - 0.7) / 0.8)
            s.entrance(lambda layer, y=y: layer.text(200, y, name, 27, MUTED), 1 - p, dx=-60, dy=0)
            if 0.7 < t < 1.5:
                s.line([(200, y + 17), (200 + font(27).getlength(name) * p, y + 17)], MUTED, 2)
                s.text(640, y + 8, "Not an emotion", 17, MUTED)


def annotate(s, t):
    s.text(64, 40, "6. Annotate features", 30)
    s.text(140, 161, "Happiness", 31)
    s.line([(140, 216), (940, 216)])
    s.entrance(
        lambda layer: (
            layer.text(140, 251, "Definition", 19, MUTED),
            layer.text(330, 247, "Feeling pleased or content.", 25),
        ),
        (t - 0.35) / 0.5,
        dx=18,
        dy=0,
    )
    s.entrance(
        lambda layer: (
            layer.text(140, 325, "Family", 19, MUTED),
            layer.text(330, 321, "Positive emotion", 25, BLUE),
        ),
        (t - 0.95) / 0.5,
        dx=18,
        dy=0,
    )


def merge(s, t):
    s.text(64, 40, "7. Merge equivalent features", 30)
    p = ease((t - 0.5) / 1.1)
    for i, name in enumerate(FEATURES[:2]):
        x = 110 + 512 * p
        y = 184 + i * 117 + (242 - 184 - i * 117) * p
        s.entrance(lambda layer, x=x, y=y, name=name: layer.text(x, y, name, 26), 1 - p, dx=0, dy=0)
    if 0 < p < 1:
        s.arrow((410, 253), (559, 253), p)
    s.entrance(
        lambda layer: (
            layer.rect((598, 221, 988, 284), fill=PALE),
            layer.text(621, 238, "Happiness", 29),
        ),
        (t - 1.05) / 0.6,
        dx=-20,
        dy=0,
    )
    retained_progress = ease((t - 0.7) / 1.0)
    s.text(110 + 511 * retained_progress, 384 - 50 * retained_progress, FEATURES[2], 26)
    if t > 1.6:
        s.entrance(lambda layer: (layer.paper(629, 397), layer.paper(676, 397)), (t - 1.6) / 0.4)


def inventory(s, t):
    s.text(64, 40, "Merged emotion inventory", 30)
    s.entrance(
        lambda layer: (
            layer.rect((100, 161, 980, 204), fill="#f5f7f9"),
            layer.text(115, 174, "Emotion", 18, MUTED),
            layer.text(401, 174, "Definition", 18, MUTED),
            layer.text(733, 174, "Papers", 18, MUTED),
            layer.text(115, 225, FEATURES[0], 24),
            layer.text(401, 222, "Feeling pleased", 22),
            layer.text(401, 253, "or content.", 22),
            layer.text(733, 230, PAPERS[0], 19, BLUE),
            layer.line([(100, 292), (980, 292)]),
            layer.text(115, 307, FEATURES[2], 24),
            layer.text(401, 304, "Feeling unhappy", 22),
            layer.text(401, 335, "or disappointed.", 22),
            layer.text(733, 309, PAPERS[0], 19, BLUE),
            layer.text(733, 340, PAPERS[1], 19, BLUE),
            layer.line([(100, 394), (980, 394)]),
        ),
        t / 0.5,
    )


SCENES = [question, collect, select, extract, filter_features, annotate, merge, inventory]


def render(time):
    scene = 0
    while scene < len(SCENES) - 1 and time >= DURATIONS[scene]:
        time -= DURATIONS[scene]
        scene += 1
    active_time = max(0, time - TRANSITION_SECONDS) if scene else time
    current = Slide()
    SCENES[scene](current, active_time / MOTION_SCALE)
    if scene and time < TRANSITION_SECONDS:
        halfway = TRANSITION_SECONDS / 2
        white = Image.new("RGB", (WIDTH, HEIGHT), "white")
        if time < halfway:
            previous = Slide()
            prior_duration = DURATIONS[scene - 1] - (TRANSITION_SECONDS if scene > 1 else 0)
            SCENES[scene - 1](previous, prior_duration / MOTION_SCALE)
            current.image = Image.blend(previous.image, white, ease(time / halfway))
        else:
            current.image = Image.blend(white, current.image, ease((time - halfway) / halfway))
    # This panel is independent of the transition: the question never fades out.
    progress = (time / MOTION_SCALE - 1.7) / 0.75 if scene == 0 else 1
    return question_panel(current.image, progress)


if __name__ == "__main__":
    DESTINATION.mkdir(parents=True, exist_ok=True)
    frames = [render(index * FRAME_MS / 1000) for index in range(round(sum(DURATIONS) * 1000 / FRAME_MS))]
    palette = Image.new("RGB", (WIDTH, HEIGHT * len(SCENES)), "white")
    elapsed = 0
    for index, duration in enumerate(DURATIONS):
        palette.paste(render(elapsed + duration - 0.4), (0, index * HEIGHT))
        elapsed += duration
    palette = palette.quantize(colors=128)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(
        DESTINATION / "workflow.gif",
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
