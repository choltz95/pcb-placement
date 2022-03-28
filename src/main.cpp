#include <stdlib.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <map>
#include <unistd.h>

using namespace std;

const clock_t start_time = clock();
constexpr float temp_th = 0.001, rej_ratio = 0.99;

inline float randf() { return float(rand())/RAND_MAX; }
inline float randb() { return rand()%2; }

typedef tuple<int, int, int> int3;

#include "floor_plan.hpp"

template<typename ID, typename LEN>
class SA {
public:
  SA(FLOOR_PLAN<ID, LEN>& fp, char** argv, ID Nblcks, int W, int H, float R,
     float P, float alpha_base,float beta)
    : _fp(fp), _Nblcks(Nblcks), _N(Nblcks), _R(R), _best_sol(fp.get_tree()),
      _alpha_base(alpha_base), _alpha(alpha_base), _N_feas(0),
      _beta(beta), _W(W), _H(H), _true_alpha(stof(argv[1])) {
    _fp.init();
    vector<int3> costs(_N+1);
    costs[0] = _fp.cost();
    _avg_r = float(get<2>(costs[0]))/get<1>(costs[0]);
    _avg_hpwl = get<0>(costs[0])/2.;
    _avg_area = get<1>(costs[0])*get<2>(costs[0]);
    _avg_true = _true_alpha*_avg_area + (1-_true_alpha)*_avg_hpwl;
    for(ID i = 1; i<=_N; ++i) {
      _fp.perturb();
      _fp.init();
      costs[i] = _fp.cost();
      const float area = get<1>(costs[i])*get<2>(costs[i]);
      const float hpwl = get<0>(costs[i])/2.;
      _avg_hpwl += hpwl;
      _avg_area += area;
      _avg_true += _true_alpha*area+(1-_true_alpha)*hpwl;
      const float r = float(get<2>(costs[i]))/get<1>(costs[i]);
      _avg_r += (r-R)*(r-R);
    }
    _fp.restore(_best_sol);
    _avg_hpwl = _avg_hpwl*1.1/(_N+1);
    _avg_area = _avg_area*1.1/(_N+1);
    _avg_r = _avg_r*1.1/(_N+1);
    _avg_true = _avg_true*1.1/(_N+1);
    float avg_cost = 0;
    for(ID i = 1; i<=_N; ++i)
      avg_cost += abs(norm_cost(costs[i]) - norm_cost(costs[i-1]));
    _init_T = -avg_cost/_N/logf(P);
  };
  pair<float, typename FLOOR_PLAN<ID, LEN>::TREE>
  run(const int k, int rnd, const float c) {
    bests.clear(), ax.clear(), Ts.clear();
    tt = clock();
    _recs.resize(_N, false);
    _N_feas = 0;
    _beta = 0.0;
    int reset_th = 2*_Nblcks, stop_th = 4*_Nblcks;
    _fp.init();
    int iter = 1, tot_feas = 0;
    float _T = _init_T, prv_cost = norm_cost(_fp.cost(_alpha, _beta));
    _best_cost = prv_cost;
    int rej_num = 0, cnt = 1;
    typename FLOOR_PLAN<ID, LEN>::TREE last_sol = _fp.get_tree();
    while(_T > temp_th || float(rej_num) <= rej_ratio*cnt) {
      if(tot_feas) _beta += 0.01;
      float avg_delta_cost = 0;
      rej_num = 0, cnt = 1;
      for(; cnt<=rnd; ++cnt) {
        _fp.perturb();
        _fp.init();
        int3 costs = _fp.cost(_alpha, _beta);
        float cost = norm_cost(costs);
        float delta_cost = (cost - prv_cost);
        avg_delta_cost += abs(delta_cost);

        if(*_recs.begin()) --_N_feas;
        _recs.pop_front();
        if(feas(costs)) {
          ++_N_feas;
          _recs.push_back(true);
          ++tot_feas;
        } else _recs.push_back(false);
        _alpha = _alpha_base + (1-_alpha_base)*_N_feas/_N;

        if(delta_cost <= 0 || randf() < expf(-delta_cost/_T) || tot_feas == 1) {
          prv_cost = cost;
          last_sol = _fp.get_tree();
          if(feas(costs)) {
            if(cost < _best_cost || tot_feas == 1) {
              _best_sol = _fp.get_tree();
              _best_cost = cost;
            }
          }
        } else {
          _fp.restore(last_sol);
          ++rej_num;
        }
      }
      ++iter;
      if(iter <= k) _T = _init_T*avg_delta_cost/cnt/iter/c;
      else _T = _init_T*avg_delta_cost/cnt/iter;
      _fp.init();
      if(!tot_feas) {
        if(iter > reset_th) {
          _T = _init_T;
          iter = 1;
          reset_th += 1;
          stop_th += 1;
          rnd += 1;
        }
      } else if(iter > stop_th) break;
    }
    _fp.restore(_best_sol);
    _fp.init();
    int3 costs = _fp.cost();
    _best_cost = true_cost(costs);
    float hpwl = get<0>(costs)/2.;
    int area = get<1>(costs)*get<2>(costs);

    cerr << "     init_T: " << _init_T << '\n';
    cerr << "temperature: " << _T << '\n';
    cerr << "       iter: " << iter << '\n';
    cerr << "success num: " << tot_feas << '\n';
    cerr << "  rej_ratio: " << float(rej_num)/cnt << '\n';
    cerr << "      alpha: " << _alpha << '\n';
    cerr << "       beta: " << _beta << '\n';
    cerr << "       hpwl: " << hpwl << '\n';
    cerr << "       area: " << area << '\n';
    cerr << " total cost: " << _best_cost << '\n';
    return {_best_cost, _best_sol};
  }
private:
  float norm_cost(const int3& cost) const {
    const float r = float(get<2>(cost))/get<1>(cost);
    return (_alpha*get<1>(cost)*get<2>(cost)/_avg_area
         + _beta*get<0>(cost)/_avg_hpwl/2.
         + (1-_alpha-_beta)*(r-_R)*(r-_R)/_avg_r);
  }
  float true_cost(const int3& cost, const float den = 1) const {
    return (_true_alpha*get<1>(cost)*get<2>(cost)
            + (1-_true_alpha)*get<0>(cost)/2.) / den;
  }
  bool feas(const int3& cost) {
    return (get<1>(cost) <= _W && get<2>(cost) <= _H);
  }
  FLOOR_PLAN<ID, LEN>& _fp;
  typename FLOOR_PLAN<ID, LEN>::TREE _best_sol;
  float _best_cost;
  const ID _Nblcks, _N;
  const int _W, _H;
  const float _alpha_base, _R, _true_alpha;
  float _alpha, _beta, _init_T, _avg_hpwl, _avg_area, _avg_r, _avg_true;
  ID _N_feas;
  list<bool> _recs;
  vector<float> bests, Ts, ax;
  clock_t tt;
};
void my_main(int argc, char** argv) {
  ifstream fblcks(argv[2], ifstream::in);
  ifstream fnets(argv[3], ifstream::in);
  string rptPost = ".rpt";
  string blksPost = ".block";
  string plPost = ".out.pl";
  ofstream outs1(argv[4]+rptPost, ifstream::out);
  ofstream outs2(argv[4]+blksPost, ifstream::out);
  ofstream outs3(argv[4]+plPost, ifstream::out);
  int Nnets, W, H, Nblcks, Ntrmns;

  string ign;
  fnets >> ign >> Nnets;
  fblcks >> ign >> W >> H;
  fblcks >> ign >> Nblcks;
  fblcks >> ign >> Ntrmns;

  // nonargs
  float R = float(H)/W, costs[2];

  float P = atof(argv[5]);
  float alpha_base = atof(argv[6]);
  float beta = atof(argv[7]);
  int kcoeff = atoi(argv[8]);
  int k = max(2, Nblcks/kcoeff);
  int rnd = atoi(argv[9])*Nblcks;
  float ccoeff = atof(argv[10]);
  float c = max(ccoeff*Nblcks, (float)10.0);

  auto fp =
    FLOOR_PLAN<short, int>(fnets, fblcks, argv, Nnets, Nblcks, Ntrmns, W, H);
  auto sa =
    SA<short, int>(fp, argv, Nblcks, W, H, R, P, alpha_base, beta);
  FLOOR_PLAN<short, int>::TREE trees[2];
  for(int i = 0; i<2; ++i) {
    tie(costs[i], trees[i]) = sa.run(k, rnd, c);
  }
  fp.restore(costs[0]<costs[1] ? trees[0] : trees[1]);
  fp.init();
  fp.output(outs1, outs2, outs3);
  outs1.close();
  outs2.close();
  outs3.close();
}
int main(int argc, char** argv) {
  ios_base::sync_with_stdio(false);
  srand(time(NULL));
  my_main(argc, argv);
}
