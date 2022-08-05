# Workshop Slides

Directory for slides of the [Simulation-based Inference workshop](https://mlcolab.org/sbi-workshop). It contains an example for `jupyter` slides with interactive elements. We strive to use these text-based slides as it makes further development simpler! 

If you have any questions regarding the template, `interact`, or `RISE`, contact [Stefan](https://github.com/wastedsummer).

These `jupyter` notebooks store all sorts of nasty metadata which makes them prone to causing merge conflicts. Thus, we recommend using `nbstripout`. This `python` library sets up `git` hooks to filter out unnecessary metadata. For security purposes, these cannot be set up from just cloning. One way to get `nbstripout` in your system is to create an environment from the bundled `environment.yml` file:
```bash
conda env create -f environment.yml # will create `sbi-workshop`
conda activate sbi-workshop         # ready to work with the slides
```

To set up the hooks, simply run
```bash
nbstripout --install
nbstripout --install --attributes .gitattributes

# test whether hook is working
nbstripout --dry-run slides/example_slides.ipynb
```
in the `sbi-workshop` directory.
