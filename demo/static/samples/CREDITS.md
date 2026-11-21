# Sample images

The demo ships four photographs so it can be tried on a device with no camera —
one per description mode. They are stand-ins for a head-mounted camera frame,
not captures from the device.

All four come from Wikimedia Commons under open licenses. Each file here is a
resized copy of the Commons original; resizing does not change the license, and
the copies below remain available under the license listed for each.

| File | Mode | Author | License | Source |
|---|---|---|---|---|
| `outdoor.jpg` | outdoor nav | ManuelB701 | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | [Hillesheim (Rheinhessen), Continuous Sidewalk](https://commons.wikimedia.org/wiki/File:Hillesheim_(Rheinhessen),_Continuous_Sidewalk.jpg) |
| `indoor.jpg` | indoor nav | w_lemay | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) | [Interior, Grove Arcade, Asheville, NC](https://commons.wikimedia.org/wiki/File:Interior,_Grove_Arcade,_Asheville,_NC_-_53418766888.jpg) |
| `text.jpg` | text reading | Peter Miranda | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | [Cafe menu (Unsplash)](https://commons.wikimedia.org/wiki/File:Cafe_menu_(Unsplash).jpg) |
| `scene.jpg` | scene | Petterin | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Myyr York Cafe](https://commons.wikimedia.org/wiki/File:Myyr_York_Cafe.jpg) |

`recorded.json` holds one saved run per sample, captured from the Jetson Orin
Nano over the real `/api/describe` path (Veris-3B-v1, Q4_K_M) — the streamed
text and its measured timings, verbatim. The web app replays these, labelled as
recorded, when the board is offline. Nothing in it is written by hand.
