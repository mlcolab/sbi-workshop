# Workshop Slides

Directory for slides of the [Simulation-based Inference workshop](mlcolab.org/sbi-workshop). It contains an example for `jupyter` slides with interactive elements. Based on this example, we will create a template in this style for every speaker. Feel free to use the template but do not feel forced to. Any kind of slide-set is welcome! If you have any questions regarding the template, `interact`, or `RISE`, contact [Stefan](https://github.com/wastedsummer).
These `jupyter` notebooks store all sorts of nasty metadata which makes them prone to causing merge conflicts. Thus, we recommend using `nbstripout`. This `python` library sets up `git` hooks to filter out unnecessary metadata. For security purposes, these cannot be set up from just cloning. The library itself is included in the accompanying `environment.yml`. To set up the hooks, simply run
```bash
nbstripout --install
nbstripout --install --attributes .gitattributes

# test whether hook is working
nbstripout --dry-run slides/example_slides.ipynb
```
in the `sbi-workshop` directory. However, you do not need to use `nbstripout`.