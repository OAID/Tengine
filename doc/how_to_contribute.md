# How to contribute to Tengine repository


## Before you start contributing you should
Make sure you agree to contribute your code under Tengine (Apache License, Version 2.0) license.

## What can I do for Tengine:

### For Users
- Star our project [Tengine](https://github.com/OAID/tengine) for more users can find us .
- Write tutorials on your blogs.
- Reporting bugs in [issue page](https://github.com/OAID/tengine/issues) if you find bugs of Tengine.
- Submit demo network (shufflenet, vgg, googlenet etc.)to our [examples](https://github.com/OAID/Tengine/tree/master/examples) to help other users.
- Help others, answer questions on [issue page](https://github.com/OAID/tengine/issues).
- Any suggestions for Tengine by [tengine_dev@openailab.com](mailto://tengine_dev@openailab.com).

### For Developers
- Follow [developer guide](https://github.com/OAID/tengine/blob/master/doc/operator_dev.md) to implement **more operators**
- Support serialization for more DL-framework (**tensorflow, mxnet, pytorch,...**)
- Help us optimize the core code for more platform: **ARMv7**
- Build Tengine lib (.a, .so etc.) for **Android/IOS** users.

## How to Pull Request:
1. [Create a your personal fork](https://help.github.com/articles/fork-a-repo/)
   of the [main Tengine repository](https://github.com/OAID/tengine) in GitHub.
2. Create your branch from master via 
    ```
    git branch mybranch
    git checkout mybranch
    ``` 
    and make your changes in your new branch.

3. [Generate a pull request](https://help.github.com/articles/creating-a-pull-request/)
   through the Web interface of GitHub.
4. General rule for code style, please follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
   
   
Once you pull request, Tengine developers will review your code. As soon as tested your pull request codes, Tengine developers will merge your pull request and mark your contribution on release announcements. 

## Contacts
Have any suggestions or questions to Tengine? Feel free to contact us by [tengine_dev@openailab.com](mailto://tengine_dev@openailab.com).


