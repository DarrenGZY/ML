function result = gammaDiv( a,d )
    temp = 0;
    for j = 1:d
        temp = temp + log(gamma(0.5*(a+1)+(1-j)/2)) - log(gamma(a/2 + (1-j)/2));
    end
    result = exp(temp);
end

