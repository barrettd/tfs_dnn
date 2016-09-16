//
//  DnnTrainerAdaGrad.hpp
//
//  Created by Barrett Davis on 9/15/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
//

#ifndef DnnTrainerAdaGrad_hpp
#define DnnTrainerAdaGrad_hpp

#include "DnnTrainer.hpp"

namespace tfs {
    
    class DnnTrainerAdaGrad : public DnnTrainer {
    protected:
        DNN_NUMERIC m_eps;
        
    public:
        DnnTrainerAdaGrad( Dnn *dnn );
        virtual ~DnnTrainerAdaGrad( void );
        
        virtual DNN_NUMERIC train( const DNN_INTEGER expectation );     // Returns loss.
        
        DNN_NUMERIC eps( void ) const ;                  // get()
        DNN_NUMERIC eps( const DNN_NUMERIC value );      // set()
        
        
    };
    
}   // namespace tfs

#endif /* DnnTrainerAdaGrad_hpp */
