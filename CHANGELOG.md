
# Change Log

## [2.0][] - 2019-06-24

### Added
- Add the changelog 
- The definition, training and post-processing configurations for FCN8, U-Net, Naylor, SegNet and Micro-Net used in the nuclei instance segmentation comparative study.
- Simultaneous nuclei instance segmentation and pixel-wise nuclei type classification capability for Micro-Net, U-Net and Naylor.

### Updated
- Major extension to the XY-Net and rename repository's name to HoVer-Net to match with paper's new title
- Major restructuring the framework to allow more dynamic parameters configuration
- Update AJI so it matches with the latest from MoNuSeg's organizer.
- Update AJI+ to use a more robust matching mechanism and deprecate its usage.
- Update Panoptic Quality calculation to support IoU threshold < 0.5

## [1.0][] - 2018-12-06
### Initial release


[Unreleased]: https://github.com/jesstelford/version-changelog/compare/v3.1.1...HEAD
[2.0]: https://github.com/jesstelford/version-changelog/tree/v1.0.0
[1.0]: https://github.com/jesstelford/version-changelog/tree/v1.0.0