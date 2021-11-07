// miniprogram/pages/center/center.js
//从数据库获取数据
Page({
  /**
   * 页面的初始数据
   */
  data: {
    userInfo: {},
    hasUserInfo: false,
    haslogin: false,
    canIUse: swan.canIUse('button.open-type.getUserInfo'),
  },
  getUserInfo(e) {
    this.setData({
        userInfo: e.detail.userInfo,
        hasUserInfo: true
    });
    console.log(e.detail)
   },

   login(e) {
    this.setData({
        userInfo: e.detail.userInfo,
        hasUserInfo: true
    });
   },
   downloadFile() {
    this.toast('正在保存', 'loading');
    swan.downloadFile({
        url: 'https://agree.pescn.top/api/download/standard.doc',
        header: {
            'content-type': 'application/json'
        },
        filePath: 'bdfile://usr/agreement3.doc',
        success: res => {
            let filePath = res.filePath;
            swan.showModal({
                title: '文件下载完成',
                content: '是否需要打开？',
                confirmText: '打开',
                success: res => {
                    if (res.confirm) {
                        swan.openDocument({
                            filePath: filePath,
                            fileType: 'pdf',
                            success: res => {
                                console.log('openDocument', res)
                            },
                            fail: err => {
                                console.log('openDocument', err)
                                this.toast('打开失败');
                            }
                        });
                    }
                }
            });
        },
        fail: err => {
            this.toast('下载文件失败');
        },
        complete: () => {
            swan.hideToast();
        }
    });
},
toast(title, icon = 'none') {
    swan.showToast({title, icon});
},
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {},

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {},

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {},

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {},

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {},

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {},

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {},

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {}
});