INIT=rf"""
source workenv/bin/activate && \\
export PYTHONPATH=$PYTHONPATH:$(pwd) && \\
cd ray_docai 

"""
key= "REDACTED_GITHUB_TOKEN"

RE_CLONE_QFCRORE =r"""

rm -rf qf_core_base && \\
git clone https://github.com/wired87/qf_core_base 

"""

RE_CLONE_G =r"""
rm -rf _google && \\
git clone https://github.com/wired87/_google 
"""

RE_CLONE_FB_CORE =r"""
rm -rf fb_core && \\
git clone https://github.com/wired87/fb_core 
"""
RE_CLONE_QF_SIM =r"""
rm -rf qf_sim && \\
git clone https://github.com/wired87/qf_sim 
"""

RE_CLONE_RAYB =r"""
rm -rf _ray_core && \\
git clone https://github.com/wired87/_ray_core 

"""
RE_CLONE_RAYB =r"""
rm -rf utils && \\
git clone https://github.com/wired87/utils 

"""