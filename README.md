![image](https://github.com/Extraltodeus/LoadLoraWithTags/assets/15731540/150f926f-6c9e-44d0-801f-7de6df9d6993)

# LoadLoraWithTags
- Save/Load trigger words for loras from a json and auto fetch them on civitai if they are missing.
- Optional prompt input to auto append them (togglable, simply returns the prompt if disabled).
- Actual alphabetical order.
- Print trigger words to terminal.
- Bypass toggle to disable without aiming the sliders at 0.
- Force fetch (not in the screenshot) if you want to force a refresh.
- It uses the sha256 hash so you can rename the file as you like.

I'm talking about these words right there:

![image](https://github.com/Extraltodeus/LoadLoraWithTags/assets/15731540/f4685bd4-5575-4055-a589-89e77eee1365)


# LoraTagsQueryOnly
- Same as LoadLoraWithTags but without the need of inputing a model and a clip

# TagsSelector
- Allow to chose which tags to select.
- Can use ranges `5:8`
- Can select individal indexes `2,3,9`
- Can mix both `2,3,1:5,9`
- Can use negative values like in a python list to start from the end `2, -5:`

# TagsViewer
- Helper to show the available tag and their indexes