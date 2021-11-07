//index.js
var util = require('../../utils/util.js');
var cache = require('../../utils/cache.js');


Page({
  data: {
    main_info: {
      'title': '离婚协议一键获取',
      'min_title': '点击右下角加号,人工智能助您完成离婚协议'
    },
    btn_name: '最新离婚协议简易标准模板',
    btn_name2: '无子女财产离婚协议简易模板',

    swiperList: [
        {
            className: 'color-a',
            value: 'A'
        }, {
            className: 'color-b',
            value: 'B'
        },
    ],
    current: 0,
    itemId: 0,
    switchIndicateStatus: true,
    switchAutoPlayStatus: false,
    switchVerticalStatus: false,
    switchDuration: 500,
    autoPlayInterval: 2000,
    slider: [{
        imageUrl: '../../images/lunbo2.png'
    }, {
        imageUrl: '../../images/lunbo1.png'
    },],
    swiperCurrent: 0,
    currentTab: 0

  },
    swiperChangeCustom(e) {
        this.setData({
            swiperCurrent: e.detail.current
        });
    },
  standard: function (event) {
    swan.navigateTo({
      url: '../standard/standard'
    });
  },
  simple: function (event) {
    swan.navigateTo({
      url: '../simple/simple'
    });
  },
  addForm: function (event) {
    swan.navigateTo({
      url: '../upload/upload'
    });
  },







  onShareAppMessage: function () {

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
  onReachBottom: function () {}
});