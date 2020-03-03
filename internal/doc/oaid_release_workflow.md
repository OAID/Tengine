# Github OAID/Tengine release workflow

There are three git repos related with Tengine

    gitlab/openailabPrivate/tengine  (restricted access)
    gitlab/OAID/Tengine              (internal access)
    github/OAID/Tengine              (public access)

This document describes the workflow to generate a github version of Tengine

## prepare oaid_release branch in tengine master repo

      step 1.  git checkout oaid_release
      step 2.  merge master
               Resolve conflicts
      step 3. review all files added/deleted carefully: 
               executor/operator/arm32 should be removed
               executor/operator/arm64  should be un-touched
               internal/ should be removed
               benchmark/ should be removed
      step 4.  a few files are common in both, but have different contents. 
               
               README.md       ---- keep the oaid ones
               doc/install.md   --- keep the oaid ones
               doc/benchmark.md  --- keep the oaid ones
      step 5.  After review and check files, commit and push oaid_release
               
      step 6.  build, and run test and fix issues
                Build settings: ARM:     arm64 and opencl
                                x86:     openblas
                                cross:   arm64
                Test:   jenkins/core_test.list
      
      step 7.  commit the tested version and push
      
      step 7.  Create a tarball by
               git archive --format=tar HEAD | gzip > oaid_release.tgz
               
## update gitlab/OAID/Tengine

      gitlab/OAID/Tengine is used for internal test and to sync on small patch set with master repo.
     

      step 1. switch to OAID/Tengine project 
                git clone  gitlab@219.139.34.186:OAID/Tengine.git

      step 2. update all files
                rm -rf *
                tar -zxvf oaid_release.tgz
                git add *

      step 3. generate and check the whole patch

                git diff > /tmp/oaid.patch

      step 4. review the patch to double check all revisions are correct
      
      step 5. build and run test on the version

      step 6. commit and push, log the major patches this patch has
               ** log the commmit id of oaid_release **


## publish in github

      Once gitlab/OAID/Tengine has accumulated enough patches, it is time to update the github repo
      
      step 0. create a tag in gitlab/OAID/Tengine 
              If necessary, ask QA team to test the version

      step 1. get the OAID/Tengine release tarball by
            git archive --format=tar HEAD | gzip > publish.tgz

      step 2. checkout github version and update all files
            git clone git@github.com:OAID/Tengine.git
            rm -rf *
            tar -zxvf publish.tgz
            git add *

      step 3. generate and check the whole patch 

             git diff > /tmp/github.patch
      
      step 4. review the patch to double check all revisions are correct
      
      step 5. build and run test on the version

      step 6. commit and push




